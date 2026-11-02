import os, torch
from diffusers import FluxPipeline

ckpt = "black-forest-labs/FLUX.1-dev"
lora_dir = "./lora_cctvParkingPose"  # adjust if saved elsewhere

pipe = FluxPipeline.from_pretrained(ckpt, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")

base = ("hooded person, feet apart, arms extended with a two-hand grip at chest height, "
        "underground parking garage at night, warm sodium lights, CCTV scanlines and HUD, cctvParkingPose")
neg  = "gun, pistol, rifle, barrel, trigger, muzzle, knife, blood, watermark, text artifact"

objects = [
    "smartphone", "wallet", "sunglasses case", "power bank with USB cable",
    "compact camera", "TV remote", "folded umbrella", "deodorant spray can",
    "key fob with dangling keys", "small water bottle", "vape pen", "hairbrush"
]

os.makedirs("gen", exist_ok=True)
for i, obj in enumerate(objects, 1):
    prompt = f"{base}, holding a {obj} with both hands"
    img = pipe(prompt, negative_prompt=neg, height=768, width=512,
               guidance_scale=3.5, num_inference_steps=34,
               generator=torch.Generator('cpu').manual_seed(100+i)).images[0]
    img.save(f"gen/cctv_{i:02d}_{obj.replace(' ','_')}.png")
