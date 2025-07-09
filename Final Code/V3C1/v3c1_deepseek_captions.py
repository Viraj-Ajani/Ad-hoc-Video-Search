#!/usr/bin/env python3
import os
import json
import tarfile
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from concurrent.futures import ThreadPoolExecutor

# === Configuration ===
INPUT_TGZ_DIR = "/path/to/tgz_files"  # Google Drive mounted path
OUTPUT_DIR = "./vl2_captions"         # Output folder
TEMP_DIR = "./temp_extracted"         # Temporary extraction
CHECKPOINT_FILE = "./checkpoint.json" # Progress tracking
MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"  # Use -small if you have A100/H100
PROMPT = "Describe this image in detail:"
BATCH_SIZE = 4  # Conservative for RTX 3090
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# Setup directories
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(TEMP_DIR).mkdir(exist_ok=True)

# Load checkpoint
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        processed = set(json.load(f))
else:
    processed = set()

# Model Loading (with 8-bit quantization for VL2-Small)
try:
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        # For VL2-Small add: load_in_8bit=True (requires bitsandbytes)
    )
    print(f"✅ Loaded {MODEL_ID} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    print("Solutions:")
    print("1. Install deepseek_vl: pip install git+https://github.com/deepseek-ai/DeepSeek-VL")
    print("2. For VL2-Small: pip install bitsandbytes")
    raise

def process_batch(image_paths, tgz_name):
    """Process images and generate captions"""
    output_folder = Path(OUTPUT_DIR) / tgz_name
    output_folder.mkdir(exist_ok=True, parents=True)
    
    try:
        # Batch load images using DeepSeek's optimized loader
        pil_images = load_pil_images(image_paths)
        inputs = processor(
            text=[PROMPT] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
        
        # Save results
        for img_path, output in zip(image_paths, outputs):
            caption = processor.decode(output, skip_special_tokens=True).split(":")[-1].strip()
            with open(output_folder / f"{Path(img_path).stem}.txt", "w") as f:
                f.write(caption)
            processed.add(str(img_path))
            
    except Exception as e:
        print(f"⚠️ Batch failed: {e}")

def process_tgz(tgz_path):
    """Process .tgz files with atomic checkpointing"""
    tgz_name = Path(tgz_path).stem
    extracted_path = Path(TEMP_DIR) / tgz_name
    
    try:
        if str(tgz_path) in processed:
            return
            
        # Extract archive
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=extracted_path)
        
        # Find all images
        image_paths = [
            str(p) for p in extracted_path.rglob("*") 
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and str(p) not in processed
        ]
        
        # Process in batches
        for i in range(0, len(image_paths), BATCH_SIZE):
            process_batch(image_paths[i:i+BATCH_SIZE], tgz_name)
            
    except Exception as e:
        print(f"❌ {tgz_path} failed: {e}")
    finally:
        # Cleanup and atomic checkpoint update
        shutil.rmtree(extracted_path, ignore_errors=True)
        temp_ckpt = f"{CHECKPOINT_FILE}.tmp"
        with open(temp_ckpt, "w") as f:
            json.dump(list(processed), f)
        os.replace(temp_ckpt, CHECKPOINT_FILE)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(f"⚙️ Device: {DEVICE} | Model: {MODEL_ID} | Batch: {BATCH_SIZE}")
    
    # Process all .tgz files
    tgz_files = sorted(Path(INPUT_TGZ_DIR).glob("*.tgz"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_tgz, tgz_files)
    
    print("✅ Caption generation complete!")
