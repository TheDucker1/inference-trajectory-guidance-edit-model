# Unofficial implementation of [Inference-time Trajectory Optimization for Manga Image Editing](https://arxiv.org/abs/2603.27790)

Tested on [FLUX.2 Klein Base 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), and [VIBE](https://huggingface.co/iitolstykh/VIBE-Image-Edit)

Some notes:
+ `diffusers` version: `0.38.0.dev0` (installed from github)
+ `transformers` version when running Klein 9B: `5.5.4`
+ `transformers` version when running VIBE: `4.57.6` (this is important, VIBE pipeline hasn't been ported to transformers >= 5 yet)
+ the `vibe_sana_pipeline.py` file use to replaced the original file, to support cpu offloading
+ the `vibe_editor.py` file use to replaced the original `editor.py` file, to enable cpu offloading

Observation:
+ For Klein 9B (and even 4B), the conditioning not helping much, as it has good editing capability out of the box
+ For VIBE (2B, so much smaller model), the conditioning helps retain global structure, however, further finetune on hyperparameters is needed as either the model to small or hasn't been trained on manga editing task yet
+ Or there is a chance I'm implementing this wrong -_-