import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import faiss
import open_clip
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_FOLDER_DIR = "/path/to/folder_data"  # Already extracted folders
OUTPUT_DIR = "./clip_embeddings"           # Output for FAISS + names
CHECKPOINT_FILE = "./clip_ckpt.json"       # Progress tracking
BATCH_SIZE = 16                             # Adjust per GPU memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup directories
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load checkpoint
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        processed = set(json.load(f))
else:
    processed = set()

# Load CLIP model + preprocessors
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

# Move model to device
model = model.to(DEVICE)
model.eval()

def process_batch(image_paths):
    """Embed a batch of images"""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = [preprocess_val(img).unsqueeze(0).to(DEVICE) for img in images]
    inputs = torch.cat(inputs, dim=0)

    with torch.no_grad():
        features = model.encode_image(inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        features = features.cpu().numpy()

    names = [Path(p).stem for p in image_paths]
    return names, features

def process_folder(folder_path):
    """Process one folder (previously extracted from a .tgz)"""
    folder_name = Path(folder_path).name
    all_names = []
    all_embs = []

    image_paths = [
        str(p) for p in Path(folder_path).rglob("*")
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
        index = faiss.IndexFlatIP(all_embs.shape[1])
        index.add(all_embs)

        faiss.write_index(index, str(Path(OUTPUT_DIR) / f"{folder_name}.faiss"))
        with open(Path(OUTPUT_DIR) / f"{folder_name}_names.json", "w") as f:
            json.dump(all_names, f)

    # Save checkpoint atomically
    with open(f"{CHECKPOINT_FILE}.tmp", "w") as f:
        json.dump(list(processed), f)
    os.replace(f"{CHECKPOINT_FILE}.tmp", CHECKPOINT_FILE)

# Main execution
if __name__ == "__main__":
    folder_paths = [str(p) for p in Path(INPUT_FOLDER_DIR).iterdir() if p.is_dir()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_folder, folder_paths)

    print("✅ CLIP embedding extraction complete!")

