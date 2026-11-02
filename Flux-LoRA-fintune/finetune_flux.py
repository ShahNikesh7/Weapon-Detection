# train_flux_lora.py
# Launches Hugging Face Accelerate to run the FLUX LoRA DreamBooth training script
# Edit the CONFIG section below, then:  python train_flux_lora.py
# Requires: accelerate, diffusers (repo cloned for the training script), torch, transformers, bitsandbytes, etc.

import os
import sys
import shutil
import subprocess
from pathlib import Path

# -----------------------------
# CONFIG — EDIT THESE DEFAULTS
# -----------------------------
PRETRAINED = "black-forest-labs/FLUX.1-dev"

## UPDATE PATH FOR DATA ###

# Path to the Parquet produced by compute_embeddings.py
DATA_DF_PATH = r"seed_cctv_pose\embeddings_cctv.parquet"
# Path to the training script from the diffusers repo:
#   diffusers/examples/research_projects/flux_lora_quantization/train_dreambooth_lora_flux_miniature.py
# If you cloned diffusers next to this project, this relative path may work:
SCRIPT_PATH = r"diffusers\examples\research_projects\flux_lora_quantization\train_dreambooth_lora_flux_miniature.py"

# Where to save your LoRA weights
OUTPUT_DIR = r".\lora_cctvParkingPose"

# Mixed precision: "bf16" (preferred on newer NVIDIA) or "fp16"
MIXED_PRECISION = "bf16"

# Training size (lower if VRAM is tight)
WIDTH, HEIGHT = 768, 512

TRAIN_BATCH_SIZE = 1
REPEATS = 1
LEARNING_RATE = 1e-4
GRAD_ACCUM_STEPS = 4
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 0
RANK = 8
MAX_TRAIN_STEPS = 1500
SEED = 0

USE_8BIT_ADAM = True
GRADIENT_CHECKPOINTING = True

# -----------------------------
# LAUNCHER (no need to edit)
# -----------------------------
def _ensure_paths():
    missing = []
    if not Path(SCRIPT_PATH).exists():
        missing.append(("SCRIPT_PATH", SCRIPT_PATH))
    if not Path(DATA_DF_PATH).exists():
        missing.append(("DATA_DF_PATH", DATA_DF_PATH))
    if missing:
        msg = ["\n[!] Missing required files/paths:"]
        for k, p in missing:
            msg.append(f"    - {k}: {p}")
        msg.append("\nFix the paths at the top of train_flux_lora.py and try again.")
        sys.exit("\n".join(msg))

def _ensure_accelerate():
    if shutil.which("accelerate") is None:
        sys.exit(
            "[!] 'accelerate' CLI not found in this environment.\n"
            "    Install it with:\n"
            "      python -m pip install -U accelerate\n"
            "    Then re-run:  python train_flux_lora.py"
        )

def build_command():
    cmd = [
        sys.executable, "-m", "accelerate", "launch",
        SCRIPT_PATH,
        "--pretrained_model_name_or_path", PRETRAINED,
        "--data_df_path", DATA_DF_PATH,
        "--output_dir", OUTPUT_DIR,
        "--mixed_precision", MIXED_PRECISION,
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--train_batch_size", str(TRAIN_BATCH_SIZE),
        "--repeats", str(REPEATS),
        "--learning_rate", str(LEARNING_RATE),
        "--gradient_accumulation_steps", str(GRAD_ACCUM_STEPS),
        "--lr_scheduler", LR_SCHEDULER,
        "--lr_warmup_steps", str(LR_WARMUP_STEPS),
        "--rank", str(RANK),
        "--max_train_steps", str(MAX_TRAIN_STEPS),
        "--seed", str(SEED),
    ]
    if USE_8BIT_ADAM:
        cmd.append("--use_8bit_adam")
    if GRADIENT_CHECKPOINTING:
        cmd.append("--gradient_checkpointing")
    return cmd

def main():
    print("==> FLUX LoRA training launcher")
    print(f"Python: {sys.executable}")
    print(f"Script : {SCRIPT_PATH}")
    print(f"Data   : {DATA_DF_PATH}")
    print(f"Outdir : {OUTPUT_DIR}")

    _ensure_paths()
    _ensure_accelerate()

    cmd = build_command()
    print("\n==> Running command:\n" + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")

    # Use shell=False with a list to avoid Windows quoting issues
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"[!] Training exited with code {proc.returncode}")
    print("\nTraining finished successfully.")

if __name__ == "__main__":
    main()
