#!/usr/bin/env python3
import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from concurrent.futures import ThreadPoolExecutor

# === Configuration ===
INPUT_FOLDER_DIR = "/path/to/tgz_files"  # Directory with extracted folders
OUTPUT_DIR = "./vl2_captions"              # Output folder
CHECKPOINT_FILE = "./checkpoint.json"      # Progress tracking
MODEL_ID = "deepseek-ai/deepseek-vl2-tiny" # Use -small if you have A100/H100
PROMPT = "Describe this image in detail:"
BATCH_SIZE = 4                             
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup directories
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load checkpoint
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        processed = set(json.load(f))
else:
    processed = set()

# Load model
try:
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        # For VL2-Small: add load_in_8bit=True if using bitsandbytes
    ).to(DEVICE)
    print(f"✅ Loaded {MODEL_ID} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

def process_batch(image_paths, folder_name):
    """Generate captions for a batch of images"""
    output_folder = Path(OUTPUT_DIR) / folder_name
    output_folder.mkdir(exist_ok=True, parents=True)

    try:
        pil_images = load_pil_images(image_paths)
        inputs = processor(
            text=[PROMPT] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)

        for img_path, output in zip(image_paths, outputs):
            caption = processor.decode(output, skip_special_tokens=True).split(":")[-1].strip()
            with open(output_folder / f"{Path(img_path).stem}.txt", "w") as f:
                f.write(caption)
            processed.add(str(img_path))

    except Exception as e:
        print(f"⚠️ Batch failed: {e}")

def process_folder(folder_path):
    """Process all images in a folder"""
    folder_name = Path(folder_path).name

    try:
        image_paths = [
            str(p) for p in Path(folder_path).rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and str(p) not in processed
        ]

        for i in range(0, len(image_paths), BATCH_SIZE):
            process_batch(image_paths[i:i+BATCH_SIZE], folder_name)

    except Exception as e:
        print(f"❌ Failed processing {folder_path}: {e}")
    finally:
        temp_ckpt = f"{CHECKPOINT_FILE}.tmp"
        with open(temp_ckpt, "w") as f:
            json.dump(list(processed), f)
        os.replace(temp_ckpt, CHECKPOINT_FILE)

# === Main Execution ===
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(f"⚙️ Device: {DEVICE} | Model: {MODEL_ID} | Batch: {BATCH_SIZE}")

    folder_paths = [str(p) for p in Path(INPUT_FOLDER_DIR).iterdir() if p.is_dir()]
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_folder, folder_paths)

    print("✅ Caption generation complete!")

