import os
import json
import tarfile
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import faiss
import shutil
import open_clip
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_TGZ_DIR = "/path/to/tgz_files"  # Data path
OUTPUT_DIR = "./clip_embeddings"      # Final output
TEMP_DIR = "./temp_extracted"         # Temporary extraction
CHECKPOINT_FILE = "./clip_ckpt.json"  # Progress tracking
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

# Load CLIP model and tokenizer using open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

# Move model to device
model = model.to(DEVICE)
model.eval()

def process_batch(image_paths):
    """Process a batch of images and return embeddings and names"""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    # Apply the appropriate preprocessing for the CLIP model
    inputs = [preprocess_val(image).unsqueeze(0).to(DEVICE) for image in images]
    inputs = torch.cat(inputs, dim=0)  # Combine all images into a single batch

    with torch.no_grad():
        # Get image features from the CLIP model
        image_features = model.encode_image(inputs)
        
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()

    # Extract image names from paths
    names = [Path(p).stem for p in image_paths]
    return names, image_features

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

        # Gather image paths to process
        image_paths = [
            str(p) for p in extracted_path.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and str(p) not in processed
        ]

        # Process images in batches
        for i in range(0, len(image_paths), BATCH_SIZE):
            names, embs = process_batch(image_paths[i:i + BATCH_SIZE])
            all_names.extend(names)
            all_embs.append(embs)
            processed.update(image_paths[i:i + BATCH_SIZE])

        if all_embs:
            # Stack all embeddings and ensure they are of type float32 for FAISS
            all_embs = np.vstack(all_embs).astype("float32")
            
            # Create a FAISS index for cosine similarity search
            index = faiss.IndexFlatIP(all_embs.shape[1])  # Use inner product (cosine similarity)
            index.add(all_embs)

            # Save the FAISS index
            faiss.write_index(index, str(Path(OUTPUT_DIR) / f"{tgz_name}.faiss"))

            # Save corresponding names of the images
            with open(Path(OUTPUT_DIR) / f"{tgz_name}_names.json", "w") as f:
                json.dump(all_names, f)

    finally:
        shutil.rmtree(extracted_path, ignore_errors=True)
        # Save progress
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(list(processed), f)


# Main execution
if __name__ == "__main__":
    tgz_files = [str(p) for p in Path(INPUT_TGZ_DIR).glob("*.tgz")]

    # Parallel processing (4 workers)
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_tgz, tgz_files)

    print("✅ Embedding extraction complete!")

