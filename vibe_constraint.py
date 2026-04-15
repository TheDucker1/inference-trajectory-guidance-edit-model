import torch
import inspect

from vibe.generative_pipeline import VIBESanaEditingPipeline
from PIL import Image
from typing import Any, Callable
import numpy as np

from dataclasses import dataclass

from loguru import logger

XLA_AVAILABLE = False


from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.pipelines.sana.pipeline_sana import SanaPipeline, retrieve_timesteps

class VIBESanaEditingPipeLineConstraint(VIBESanaEditingPipeline):

    def _run_denoising_loop(  # noqa: PLR0913
        self,
        latents: torch.Tensor,
        latent_channels: int,
        input_image_latents: torch.Tensor | None,
        timesteps: torch.Tensor,
        num_inference_steps: int,
        num_warmup_steps: int,
        prompt_embeds: torch.Tensor,
        empty_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        image_guidance_scale: float,
        extra_step_kwargs: dict,
        *,
        is_t2i: bool = False,
        constraint_step: int = 3,
        constraint_alpha: float = 0.01
    ) -> torch.Tensor:
        """Run the denoising loop over the given timesteps.

        Args:
            latents (torch.Tensor): Latents to denoise.
            latent_channels (int): The number of latent channels.
            input_image_latents (torch.Tensor | None): The input image latents or None if t2i generation is enabled.
            timesteps (torch.Tensor): The timesteps for the scheduler.
            num_inference_steps (int): The number of total inference steps.
            num_warmup_steps (int): The number of warmup steps for the scheduler.
            prompt_embeds (torch.Tensor): The prompt embeds from edit head.
            guidance_scale (float): The guidance scale for the text prompt.
            image_guidance_scale (float): The guidance scale for the image prompt.
            extra_step_kwargs (dict): The extra step kwargs for the scheduler.
            is_t2i (bool): Whether the t2i generation is enabled.

        Returns:
            torch.Tensor: The final latents after the denoising loop.
        """
        transformer_dtype = self.transformer.dtype
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if self.do_image_guidance else 2))
                else:
                    latent_model_input = latents

                # concat noised and input image latents.
                if input_image_latents is None:  # In case of t2i generation, we don't have input image latents
                    scaled_latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=1)
                else:
                    scaled_latent_model_input = torch.cat([latent_model_input, input_image_latents], dim=1)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep * self.transformer.config.timestep_scale

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=scaled_latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                    t2i_samples=[is_t2i] * scaled_latent_model_input.shape[0],
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    if self.do_image_guidance:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_text - noise_pred_image)
                            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                    else:
                        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                if i < constraint_step:
                    B = latents.shape[0]
                    if input_image_latents is None:
                        cond_latents = latents
                        constraint_input = torch.cat([latents, latents], dim=1)
                    else:
                        cond_latents = input_image_latents[:B]
                        constraint_input = torch.cat([latents, cond_latents], dim=1)
                    
                    constraint_prompt = empty_prompt_embeds[:B]

                    constraint_noise_pred = self.transformer(
                        hidden_states=constraint_input.to(dtype=transformer_dtype),
                        encoder_hidden_states=constraint_prompt,
                        timestep=timestep[:B],
                        return_dict=False,
                        attention_kwargs=self.attention_kwargs,
                        t2i_samples=[is_t2i] * B
                    )[0]

                    if self.transformer.config.out_channels // 2 == latent_channels:
                        constraint_noise_pred = constraint_noise_pred.chunk(2, dim=1)[0]

                    delta_t = t.item() / 1000
                    correction_target_latents = cond_latents - (delta_t * constraint_noise_pred)
                    latents = (1.0 - constraint_alpha) * latents + constraint_alpha * correction_target_latents

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        return latents

    @torch.no_grad()
    def __call__(  # noqa: PLR0913,PLR0912,PLR0915
        self,
        conditioning_image: Image.Image | None = None,
        prompt: str | list[str] | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        latents: torch.FloatTensor | None = None,
        height: int | None = None,
        width: int | None = None,
        eta: float = 0.0,
        num_inference_steps: int = 20,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        image_guidance_scale: float = 1.2,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        *,
        return_dict: bool = True,
        use_resolution_binning: bool = True,
        constraint_step: int = 3,
        constraint_alpha: float = 0.01,
        **_: Any,
    ) -> SanaPipelineOutput | tuple:
        """Function invoked when calling the pipeline for generation.

        Args:
            conditioning_image (Image.Image | None): The input conditioning image or None if t2i generation is enabled.
            prompt (str | list[str] | None): The editing prompt.
            prompt_embeds (torch.FloatTensor): The prompt embeds from edit head.
            negative_prompt_embeds (torch.FloatTensor): The negative prompt embeds from edit head.
            latents (Optional[torch.FloatTensor]): The latents to use for the denoising process.
            height (Optional[int]): The height of the conditioning_image.
            width (Optional[int]): The width of the conditioning_image.
            num_inference_steps (int): The number of inference steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            guidance_scale (float): The guidance scale for the text prompt.
            image_guidance_scale (float): The guidance scale for the conditioning_image.
            num_images_per_prompt (int): The number of images to generate per prompt.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]]): The generator.
            output_type (str): The output type.
            attention_kwargs (dict[str, Any]): The attention kwargs.
            return_dict (bool): Whether to return a return_dict.
            use_resolution_binning (bool): Whether to use resolution binning.
            _: Additional keyword arguments.

        Returns:
            SanaPipelineOutput: The output of the pipeline.
        """
        # check if we need to fallback to t2i generation
        is_t2i = False
        if conditioning_image is None:
            is_t2i = True
            logger.warning("Fallback to t2i generation because conditioning_image is not provided.")

        # 0. Set pipeline attributes.
        device = self._execution_device
        dtype = self.dtype  # type: ignore
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = 0 if is_t2i else image_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if (height is None or width is None) and conditioning_image is None:
            msg = "Either height and width or conditioning_image must be provided."
            raise ValueError(msg)

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if height is None or width is None:
                height, width = conditioning_image.height, conditioning_image.width  # type: ignore[union-attr]
            orig_height, orig_width = height, width
            height, width = self.get_closest_size(height, width)

        self.check_inputs(
            prompt,
            height,
            width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Identify batch size.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]  # type: ignore

        # 1. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(  # type: ignore[assignment]
            batch_size=batch_size,
            conditioning_image=conditioning_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
        )


        empty_prompt_embeds, _ = self.encode_prompt(
            batch_size=batch_size,
            conditioning_image=None,
            prompt="",
            prompt_embeds=None,
            negative_prompt_embeds=None,
            num_images_per_prompt=num_images_per_prompt
        )

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        empty_prompt_embeds = empty_prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        # 2. Prepare conditioning image latents
        if is_t2i:
            assert False, "Custom pipeline for editing, requires image"
            image_latents = None
        else:
            processed_image = self.image_processor.preprocess(conditioning_image, height=height, width=width)
            processed_image = processed_image.to(device).to(dtype)
            image_latents = self.prepare_image_latents(
                processed_image,
                batch_size,
                num_images_per_prompt,
                self.dtype,
                device,
            )

        # 3. Prepare latents
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(  # type: ignore
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        num_channels_latents_check = (
            latent_channels * 2 if self.transformer.input_condition_type == "channel_cat" else latent_channels
        )
        if num_channels_latents_check != self.transformer.config.in_channels:
            msg = f"The config of `pipeline.transformer` expects {self.transformer.config.in_channels} channels,"
            f"but received {num_channels_latents_check}."
            raise ValueError(msg)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)  # type: ignore[arg-type]
        self._num_timesteps = len(timesteps)  # type: ignore[arg-type]

        # 5. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6. Denoising loop
        latents = self._run_denoising_loop(  # type: ignore
            latents=latents,  # type: ignore[arg-type]
            latent_channels=latent_channels,
            input_image_latents=image_latents,
            timesteps=timesteps,  # type: ignore[arg-type]
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            prompt_embeds=prompt_embeds,  # type: ignore[arg-type]
            empty_prompt_embeds=empty_prompt_embeds,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            is_t2i=is_t2i,
            constraint_step=constraint_step,
            constraint_alpha=constraint_alpha,
        )

        if output_type == "latent":
            image = latents  # type: ignore[assignment]
        else:
            latents = latents.to(self.dtype)  # type: ignore
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

        if output_type != "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)  # type: ignore[arg-type]

import random

import numpy as np
import torch

from vibe.utils import retry_decorator
from vibe.utils.img_utils import get_multiscale_transform, postprocess_padded_image, revert_resize

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, *, randomize_seed: bool = False) -> int:
    """Randomize the seed.

    Args:
        seed (int): Seed.
        randomize_seed (bool): Whether to randomize the seed.

    Returns:
        int: Randomized seed.
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)  # noqa: S311
    return seed

class VibeImageEditorConstraint:
    """Image editor class."""
    def __init__(
        self,
        checkpoint_path: str,
        image_guidance_scale: float = 1.2,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 20,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """Initialize the image editor.

        Args:
            checkpoint_path (str): The path to the local model checkpoint.
            image_guidance_scale (float): The image guidance scale.
            guidance_scale (float): The guidance scale.
            num_inference_steps (int): The number of inference steps.
            device (str): The device to use.
        """
        self.cfg_distilled = False
        self.weight_dtype = torch.bfloat16
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.get_generation_pipe(checkpoint_path, **kwargs)
        if self.cfg_distilled:
            self.image_guidance_scale = 0.0
            self.guidance_scale = 0.0
            logger.info(f"CFG-distilled model loaded. CFG is disabled by default.")

    @retry_decorator(logger=logger, delay=1)
    def get_generation_pipe(self, checkpoint_path: str, **kwargs) -> None:
        """Get the generation pipe.

        Args:
            checkpoint_path (str): The path to the pipeline checkpoint.
        """
        self.pipe = VIBESanaEditingPipeLineConstraint.from_pretrained(
            checkpoint_path,
            **kwargs
        )
        #self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()
        self.cfg_distilled = getattr(self.pipe.transformer.config, "cfg_distilled", False)

    def prepare_image_for_diffusion(self, image: Image.Image) -> tuple[Image.Image, tuple[int, ...] | None]:
        """Prepare the image for diffusion.

        Args:
            image (Image.Image): The input image.

        Returns:
            tuple[Image.Image, tuple[int, ...] | None]: The prepared image and the crop coordinates.
        """
        inp_image = image.copy()
        ori_h, ori_w = inp_image.height, inp_image.width

        # multiscale when we need to calculate the closest aspect ratio and resize & crop image
        closest_size = self.pipe.get_closest_size(ori_h, ori_w)

        transform, crop_coords = get_multiscale_transform(
            closest_size=closest_size,
            orig_size=(ori_h, ori_w),
            resize_factor=self.pipe.vae_scale_factor,
            out_img_type="pil",
        )
        inp_image = transform(inp_image)
        return inp_image, crop_coords

    @torch.inference_mode()
    def generate_edited_image(  # noqa: PLR0913
        self,
        instruction: str | list[str],
        *,
        conditioning_image: Image.Image | None = None,
        randomize_seed: bool = False,
        do_revert_resize: bool = True,
        seed: int = 42,
        num_images_per_prompt: int = 1,
        t2i_height: int | None = None,
        t2i_width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        image_guidance_scale: float | None = None,
        constraint_step: int = 3,
        constraint_alpha: float = 0.01,
    ) -> list[Image.Image]:
        """Generate an image based on the provided image and text prompt.

        Args:
            instruction (str | list[str]): Generation instruction or list of instructions.
            conditioning_image (Image.Image | None): Input image to edit or None if t2i generation is enabled.
            randomize_seed (bool): Whether to randomize the seed.
            do_revert_resize (bool): resize generated image to initial size.
            seed (int): Seed for random number generation. Defaults to 0.
            num_images_per_prompt (int): number of images to generate for each prompt
            t2i_height (int | None): The height of the t2i image.
            t2i_width (int | None): The width of the t2i image.
            image_guidance_scale (float | None): The image guidance scale.
            guidance_scale (float | None): The guidance scale.
            num_inference_steps (int | None): The number of inference steps.

        Returns:
            list[Image.Image]: The edited images.
        """
        # Randomize seed
        seed = randomize_seed_fn(seed, randomize_seed=randomize_seed)

        # Prepare input image for generation
        if conditioning_image is not None:
            orig_height, orig_width = conditioning_image.height, conditioning_image.width
            inp_image, crop_coords = self.prepare_image_for_diffusion(conditioning_image)
            inp_height, inp_width = inp_image.size[1], inp_image.size[0]
        else:
            if t2i_height is None or t2i_width is None:
                logger.warning("Height and width for t2i generation are not provided, using default value 1024x1024")
                t2i_height = t2i_width = 1024
            inp_height, inp_width = self.pipe.get_closest_size(t2i_height, t2i_width)
            if (inp_height, inp_width) != (t2i_height, t2i_width):
                logger.warning(
                    f"The desired size of t2i generation {t2i_height}x{t2i_width} "
                    f"were adjusted to {inp_height}x{inp_width} to follow default binning scheme."
                )

        # Generate edited image
        generated_images = self.pipe(  # type: ignore
            conditioning_image=conditioning_image,
            prompt=instruction,
            height=inp_height,
            width=inp_width,
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=self.guidance_scale if guidance_scale is None else guidance_scale,
            image_guidance_scale=self.image_guidance_scale if image_guidance_scale is None else image_guidance_scale,
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(seed),
            constraint_step=constraint_step,
            constraint_alpha=constraint_alpha,
        ).images

        # Return generated images if t2i generation is enabled
        if conditioning_image is None:
            return generated_images  # type: ignore[return-value]

        # remove paddings in case of letterbox usage
        generated_images = [postprocess_padded_image(im, crop_coords) for im in generated_images]  # type: ignore

        if do_revert_resize:
            return [revert_resize(im, (orig_height, orig_width)) for im in generated_images]
        return generated_images
