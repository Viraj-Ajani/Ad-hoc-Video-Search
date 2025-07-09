import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import faiss
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_FOLDER_DIR = "/path/to/folders"       # Now points to extracted folders
OUTPUT_DIR = "./llava_embeddings"           # Final output
CHECKPOINT_FILE = "./llava_ckpt.json"       # Progress tracking
BATCH_SIZE = 16                             
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
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = vision_tower(inputs)
        embs = outputs.last_hidden_state.mean(dim=1)
        embs = torch.nn.functional.normalize(embs, dim=1)
        embs = embs.cpu().numpy()

    names = [Path(p).stem for p in image_paths]
    return names, embs

def process_folder(folder_path):
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

    # Save updated checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(processed), f)


# Main execution
if __name__ == "__main__":
    folder_paths = [str(p) for p in Path(INPUT_FOLDER_DIR).iterdir() if p.is_dir()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_folder, folder_paths)

    print("✅ Embedding extraction complete for all folders!")

