import inspect
import torch

from diffusers import LongCatImageEditPipeline

import PIL
from typing import Any, Callable
import numpy as np

from dataclasses import dataclass
import math

XLA_AVAILABLE = False
from diffusers.utils import BaseOutput

@dataclass
class LongCatImagePipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: list[PIL.Image.Image, np.ndarray]


# Copied from diffusers.pipelines.longcat_image.pipeline_longcat_image.split_quotation
def split_quotation(prompt, quote_pairs=None):
    """
    Implement a regex-based string splitting algorithm that identifies delimiters defined by single or double quote
    pairs. Examples::
        >>> prompt_en = "Please write 'Hello' on the blackboard for me." >>> print(split_quotation(prompt_en)) >>> #
        output: [('Please write ', False), ("'Hello'", True), (' on the blackboard for me.', False)]
    """
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote = []

    for i, word_src in enumerate(set(matches_word_internal_quote_pattern)):
        word_tgt = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
    pattern = "|".join([re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2) for q1, q2 in quote_pairs])
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


# Copied from diffusers.pipelines.longcat_image.pipeline_longcat_image.prepare_pos_ids
def prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=None, height=None, width=None):
    if type == "text":
        assert num_token
        if height or width:
            print('Warning: The parameters of height and width will be ignored in "text" type.')
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif type == "image":
        assert height and width
        if num_token:
            print('Warning: The parameter of num_token will be ignored in "image" type.')
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(f'Unknow type {type}, only support "text" or "image".')
    return pos_ids


# Copied from diffusers.pipelines.longcat_image.pipeline_longcat_image.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = width if width % 16 == 0 else (width // 16 + 1) * 16
    height = height if height % 16 == 0 else (height // 16 + 1) * 16

    width = int(width)
    height = int(height)

    return width, height

class LongCatImageEditPipelineConstraint(LongCatImageEditPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        constraint_step: int = 3,
        constraint_alpha: float = 0.01,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Examples:

        Returns:
            [`~pipelines.LongCatImagePipelineOutput`] or `tuple`: [`~pipelines.LongCatImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] * 1.0 / image_size[1])

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            calculated_height,
            calculated_width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = self.image_processor.resize(image, calculated_height // 2, calculated_width // 2)
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)

        negative_prompt = "" if negative_prompt is None else negative_prompt
        (prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt, image=prompt_image, prompt_embeds=prompt_embeds, num_images_per_prompt=num_images_per_prompt
        )
        if self.do_classifier_free_guidance:
            (negative_prompt_embeds, negative_text_ids) = self.encode_prompt(
                prompt=negative_prompt,
                image=prompt_image,
                prompt_embeds=negative_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
            )
        (empty_prompt_embeds, empty_text_ids) = self.encode_prompt(
            prompt="", image=prompt_image, prompt_embeds=None, num_images_per_prompt=num_images_per_prompt
        )

        # 4. Prepare latent variables
        num_channels_latents = 16
        latents, image_latents, latents_ids, image_latents_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            calculated_height,
            calculated_width,
            prompt_embeds.dtype,
            prompt_embeds.shape[1],
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if image is not None:
            latent_image_ids = torch.cat([latents_ids, image_latents_ids], dim=0)
        else:
            latent_image_ids = latents_ids

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred_text = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    noise_pred_text = noise_pred_text[:, :image_seq_len]
                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
                        noise_pred_uncond = noise_pred_uncond[:, :image_seq_len]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_text

                if i < constraint_step:
                    with self.transformer.cache_context("cond"):
                        noise_pred_empty = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            encoder_hidden_states=empty_prompt_embeds,
                            txt_ids=empty_text_ids,
                        img_ids=latent_image_ids,
                            return_dict=False
                        )[0]

                        noise_pred_empty = noise_pred_empty[:, :image_seq_len]
                        delta_t = t.item() / 1000

                        correction_target_latents = image_latents - (delta_t * noise_pred_empty)
                        latents = (1.0 - constraint_alpha) * latents + constraint_alpha * correction_target_latents

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, calculated_height, calculated_width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return LongCatImagePipelineOutput(images=image)
