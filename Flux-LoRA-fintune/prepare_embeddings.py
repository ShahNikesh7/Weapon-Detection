# prepare_embeddings.py
# End-to-end helper to compute text embeddings for FLUX QLoRA using the official script.
# What it does:
#   - (a) clone Hugging Face diffusers repo if missing
#   - (b) install its research project requirements
#   - (c) run compute_embeddings.py to create a Parquet of text embeddings
#
# Usage:
#   python prepare_embeddings.py
#
# After it finishes, you'll have a file like:
#   <OUT_PARQUET>  (e.g., seed_cctv_pose/embeddings_cctv.parquet)
#
# Then point your training launcher to --data_df_path=<OUT_PARQUET>

import sys, subprocess, shutil, os
from pathlib import Path

# ------------- CONFIG (EDIT THESE) -------------
IMAGES_DIR   = r"C:\path\to\seed_cctv_pose\images"
CAPTIONS_CSV = r"C:\path\to\seed_cctv_pose\captions.csv"
OUT_PARQUET  = r"C:\path\to\seed_cctv_pose\embeddings_cctv.parquet"

# Where to put/expect the diffusers repo
DIFFUSERS_DIR = Path.cwd() / "diffusers"

# Subfolder containing the FLUX QLoRA scripts
PROJECT_DIR = DIFFUSERS_DIR / "examples" / "research_projects" / "flux_lora_quantization"

# Optional: speed up HF downloads on Windows (no symlink warnings)
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ------------- UTILITIES -------------
def run(cmd, cwd=None):
    print("\n==> ", " ".join(cmd) if isinstance(cmd, list) else cmd)
    result = subprocess.run(cmd, cwd=cwd, shell=False)
    if result.returncode != 0:
        raise SystemExit(f"[!] Command failed with code {result.returncode}")

def ensure_git():
    if shutil.which("git") is None:
        raise SystemExit("[!] 'git' is not on PATH. Install Git for Windows and retry.")

def ensure_repo():
    if DIFFUSERS_DIR.exists():
        print(f"[i] Using existing repo: {DIFFUSERS_DIR}")
        return
    ensure_git()
    run(["git", "clone", "--depth", "1", "https://github.com/huggingface/diffusers", str(DIFFUSERS_DIR)])

def ensure_requirements():
    req = PROJECT_DIR / "requirements.txt"
    if not req.exists():
        raise SystemExit(f"[!] requirements.txt not found: {req}")
    # Install requirements for the research project into the *current* interpreter/venv
    run([sys.executable, "-m", "pip", "install", "-r", str(req)])

def sanity_check_inputs():
    missing = []
    if not Path(IMAGES_DIR).exists():
        missing.append(("IMAGES_DIR", IMAGES_DIR))
    if not Path(CAPTIONS_CSV).exists():
        missing.append(("CAPTIONS_CSV", CAPTIONS_CSV))
    out_parent = Path(OUT_PARQUET).parent
    out_parent.mkdir(parents=True, exist_ok=True)
    if missing:
        msg = ["\n[!] Missing inputs:"]
        for k, p in missing:
            msg.append(f"    - {k}: {p}")
        raise SystemExit("\n".join(msg))

def compute_embeddings():
    script = PROJECT_DIR / "compute_embeddings.py"
    if not script.exists():
        raise SystemExit(f"[!] compute_embeddings.py not found: {script}")

    cmd = [
        sys.executable, str(script),
        "--images_dir", IMAGES_DIR,
        "--captions_csv", CAPTIONS_CSV,
        "--out", OUT_PARQUET,
    ]
    run(cmd)

def main():
    print("== FLUX QLoRA: Prepare text embeddings ==")
    print("Python:", sys.executable)
    print(f"Images   : {IMAGES_DIR}")
    print(f"Captions : {CAPTIONS_CSV}")
    print(f"Out file : {OUT_PARQUET}")

    sanity_check_inputs()
    ensure_repo()
    ensure_requirements()
    compute_embeddings()

    print(f"\n✅ Done. Parquet created at:\n   {OUT_PARQUET}")
    print("Next: use this path as --data_df_path in your LoRA training script.")

if __name__ == "__main__":
    main()
