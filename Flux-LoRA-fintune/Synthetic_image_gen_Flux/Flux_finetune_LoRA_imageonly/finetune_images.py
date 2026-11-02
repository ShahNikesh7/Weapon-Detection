'''
Author: Nikesh Shah

This script finetunes Flux using LoRA. We do not use text-embedding
Make sure to read the README first
'''
import os, sys, subprocess, shlex
from pathlib import Path

BASE = Path("/lambda/nfs/Flux")
TRAIN_SCRIPT = BASE / "diffusers" / "examples" / "advanced_diffusion_training" / "train_dreambooth_lora_flux_advanced.py"

PROJECT_DIR = BASE / "seed_cctv_pose"
IMAGES_DIR  = PROJECT_DIR / "images"     # contains all the finetune images only 
OUT_DIR     = PROJECT_DIR / "flux_pose_lora"
LOG_DIR     = OUT_DIR / "logs"
ACCEL_CFG   = PROJECT_DIR / "accelerate_config.yaml"

mixed_precision = "bf16"
try:
    import torch
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        mixed_precision = "fp16"
except Exception:
    mixed_precision = "fp16"

# write a simple accelerate config if you don't already have one
ACCEL_CFG.write_text(f"""compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
mixed_precision: {mixed_precision}
num_processes: 1
gpu_ids: '0'
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
use_cpu: false
""", encoding="utf-8")

OUT_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "accelerate.commands.launch",
    "--config_file", str(ACCEL_CFG),
    str(TRAIN_SCRIPT),

    "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev",

    # still required by the script API (used for validation/fallback)
    "--instance_prompt", "cctv pose subject",
    "--instance_data_dir", str(IMAGES_DIR),
    "--output_dir", str(OUT_DIR),
    "--logging_dir", str(LOG_DIR),

    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--gradient_checkpointing",
    "--learning_rate", "1e-4",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",

    "--rank", "4",
    "--max_train_steps", "2000",
    "--checkpointing_steps", "200",

    "--seed", "42",
    "--mixed_precision", mixed_precision,
    "--dataloader_num_workers", "2",
    "--report_to", "tensorboard",
    "--lora_layers", "attn.to_k,attn.to_q,attn.to_v,attn.to_out.0",
    "--guidance_scale", "1",
]

print("Launching:\n", " ".join(shlex.quote(str(c)) for c in cmd))
subprocess.run(cmd, check=True)