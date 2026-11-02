'''
Use this script to test if the your environment has been set up to communicate with Flux correctly
'''

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16  # change to torch.float16 if BF16 unsupported
)
pipe.enable_model_cpu_offload()

img = pipe(
    "a cat holding a sign that says hello world",
    height=480, width=720, guidance_scale=3.5, num_inference_steps=50,
    generator=torch.Generator('cpu').manual_seed(0)
).images[0]

img.save("flux-test.png")
print("Saved flux-test.png")