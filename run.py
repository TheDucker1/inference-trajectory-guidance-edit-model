import torch
import PIL.Image

device = "cuda"
dtype = torch.bfloat16

def resize_max_size(image, max_size=1024):
    """
    Resizes an image so the longest side is max_size, 
    maintaining aspect ratio and ensuring dimensions are divisible by 16.
    """
    w, h = image.size
    
    # Calculate scaling factor
    stack_scale = max_size / max(w, h)
    
    # Calculate new dimensions (rounded to nearest multiple of 16)
    new_w = int(round((w * stack_scale) / 32) * 32)
    new_h = int(round((h * stack_scale) / 32) * 32)
    
    # Resize image
    resized_img = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    
    return new_h, new_w, resized_img

import sys
if len(sys.argv) < 2:
    print("Usage: [script] [file]")
    exit()

img = PIL.Image.open(sys.argv[1])
HH, WW, image = resize_max_size(img, 1024)
image.save("output_unchanged.png")

#pipe = Flux2KleinPipelineConstraint.from_pretrained(
#    "black-forest-labs/FLUX.2-klein-base-4B", 
#    torch_dtype=dtype
#)
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    repo_id="iitolstykh/VIBE-Image-Edit",
    repo_type="model",
)

from vibe_constraint import VibeImageEditorConstraint

editor = VibeImageEditorConstraint(
    checkpoint_path=model_path,
    image_guidance_scale=1.0,
    torch_dtype=dtype
)

#from diffusers import LongCatImageEditPipeline
#pipe = LongCatImageEditPipeline.from_pretrained("meituan-longcat/LongCat-Image-Edit", torch_dtype=dtype)
#pipe.to(device)
#pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
#pipe.enable_sequential_cpu_offload()

#prompt="Remove the white light obstructing. Fill in the details. Man kneel placing hand on body. Keep old movie ish color style."
#prompt = ""
prompt="Remove all text. Keep everything else as is."

#image2 = pipe(image=image,
#    prompt=prompt,
#    width=WW,
#    height=HH,
#    guidance_scale=4.5,
#    num_inference_steps=20,
#    generator=torch.Generator(device=device).manual_seed(2),
#    constraint_alpha=0.1,
#).images[0]
image2 = editor.generate_edited_image(
    instruction=prompt,
    conditioning_image=image,
    num_images_per_prompt=1,
    constraint_step=3,
    constraint_alpha=0.01,
)[0]
image2.save("output_constraint.png")
exit(0)
del image2
gc.collect()
torch.cuda.empty_cache()

#image3 = pipe(image=image,
#    prompt=prompt,
#    width=WW,
#    height=HH,
#    guidance_scale=5.0,
#    num_inference_steps=20,
#    generator=torch.Generator(device=device).manual_seed(2),
#    constraint_step=0,
#).images[0]
image3 = editor.generate_edited_image(
    instruction=prompt,
    conditioning_image=image,
    num_images_per_prompt=1,
    constraint_step=0,
)[0]
image3.save("output_noconstraint.png")
