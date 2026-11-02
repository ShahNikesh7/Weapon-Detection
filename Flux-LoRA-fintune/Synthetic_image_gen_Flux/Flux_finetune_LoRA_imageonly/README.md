# Fine-Tuning a Flux Model with LoRA

This README explians how to finetune a Flux image model using LoRA. This is the IMAGE_ONLY version where we will be using a folder that has all the images we need for finetuning. 

## 1. Requirements

- GPU: NVIDIA GPU with ≥30 GB VRAM recommended.
- CUDA: 11.8+ (or a recent CUDA-compatible PyTorch).
- Python: 3.10–3.11.
- Disk: Enough space for images, checkpoints, and logs.

## 2. Environment Setup

Install the dependencies 

```bash
pip install -U transformers torch accelerate bitsandbytes datasets safetensors pillow sentencepiece peft
python -m pip install -U "git+https://github.com/huggingface/diffusers.git"
git clone https://github.com/huggingface/diffusers
```

Connect to hugging face to be able to use the Flux model. 
```bash
huggingface-cli login
```
This will ask for you unique key that you will create through your hugging face account. 

Before you use Flux, make sure to go into the huggingface directly and accept the terms and conditions. Use this link: https://huggingface.co/black-forest-labs/FLUX.1-dev

## 3. Test Flux

We need to test if we have access to flux now, run the testflux.py to generate a sample image.

## 4. Update and run the finetune script

Update the file paths in finetune_images.py to match what you have in your directory.

## 5. Generate the images

Open gen.py to add your negatives and prompts. Then run that script to generate your images.

**Tips:**

- Make sure the images folder only has the images and the ymal file that you generated.
- Make sure all the requirements are installed.
- If running out of memory in CUDA decrease resolution, batch size, and rank