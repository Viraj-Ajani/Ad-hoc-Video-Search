import os
import json
import tarfile
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import faiss
import shutil
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_TGZ_DIR = "/path/to/tgz_files" # Data path
OUTPUT_DIR = "./llava_embeddings"     # Final output
TEMP_DIR = "./temp_extracted"         # Temporary extraction
CHECKPOINT_FILE = "./llava_ckpt.json" # Progress tracking
BATCH_SIZE = 16                       # RTX 3090 can handle this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup directories
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(TEMP_DIR).mkdir(exist_ok=True)

# Load checkpoint
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        processed = set(json.load(f))
else:
    processed = set()

# Load model (optimized for 24GB VRAM)
model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
).to(DEVICE)
model.eval()

# Get vision encoder
vision_tower = model.get_vision_tower()
vision_tower.to(DEVICE)
vision_tower.eval()

def process_batch(image_paths):
    """Process a batch of images and return embeddings and names"""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = vision_tower(inputs)
        embs = outputs.last_hidden_state.mean(dim=1)  # [B, D]
        embs = torch.nn.functional.normalize(embs, dim=1)  # Normalize for cosine sim
        embs = embs.cpu().numpy()

    names = [Path(p).stem for p in image_paths]
    return names, embs

def process_tgz(tgz_path):
    """Handle one .tgz file and write FAISS index"""
    tgz_name = Path(tgz_path).stem
    extracted_path = Path(TEMP_DIR) / tgz_name

    if not tarfile.is_tarfile(tgz_path):
        return

    all_names = []
    all_embs = []

    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=extracted_path)

        image_paths = [
            str(p) for p in extracted_path.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and str(p) not in processed
        ]

        for i in range(0, len(image_paths), BATCH_SIZE):
            names, embs = process_batch(image_paths[i:i+BATCH_SIZE])
            all_names.extend(names)
            all_embs.append(embs)
            processed.update(image_paths[i:i+BATCH_SIZE])

        if all_embs:
            all_embs = np.vstack(all_embs).astype("float32")
            index = faiss.IndexFlatIP(all_embs.shape[1])  # Use inner product = cosine if normalized
            index.add(all_embs)

            faiss.write_index(index, str(Path(OUTPUT_DIR) / f"{tgz_name}.faiss"))

            # Save corresponding names
            with open(Path(OUTPUT_DIR) / f"{tgz_name}_names.json", "w") as f:
                json.dump(all_names, f)

    finally:
        shutil.rmtree(extracted_path, ignore_errors=True)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(list(processed), f)


# Main execution
if __name__ == "__main__":
    tgz_files = [str(p) for p in Path(INPUT_TGZ_DIR).glob("*.tgz")]
    
    # Parallel processing (4 workers)
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_tgz, tgz_files)
    
    print("✅ Embedding extraction complete!")
