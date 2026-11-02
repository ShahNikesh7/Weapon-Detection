'''
Author: Nikesh Shah
This script generates images using the LoRA finetuned Flux weights
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline

# --------------- CONFIG ---------------
BASE_ID    = "black-forest-labs/FLUX.1-dev"
LORA_DIR  = Path("seed_cctv_pose/flux_pose_lora/checkpoint") ## UPDATE FILEPATH TO POINT TO checkpoint
LORA_FILE = "pytorch_lora_weights.safetensors"
OUT       = Path("gen"); OUT.mkdir(exist_ok=True)

NEG = "text watermark, logo," ## INSERT ADDITIONAL NEGATIVES THAT SHOULD NOT BE IN THE GENERATED IMAGE

CAPTIONS: List[str] = [
    ## INSERT PROMPTS HERE
    "prompt_1",
    "prompt_2"
    ]

def main():
    pipe = FluxPipeline.from_pretrained(BASE_ID, torch_dtype=torch.float16).to("cuda")
    pipe.load_lora_weights(LORA_DIR, weight_name=LORA_FILE, adapter_name="pose")
    for i, pmt in enumerate(CAPTIONS, 1):
        prompt = pmt
        g = torch.Generator(device="cuda").manual_seed(100 + i)
        image = pipe(prompt, negative_prompt=NEG, height=480, width=720,
                     guidance_scale=3.5, num_inference_steps=28, generator=g).images[0]
        image.save(OUT / f"cctv_{i:02d}.png")

if __name__ == "__main__":
    main()
