import os
import tarfile
import shutil
import time
import paramiko
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# --- SFTP Configuration ---
SFTP_HOST = "ftp.itec.aau.at"
SFTP_PORT = 22
SFTP_USERNAME = "v3c"
SFTP_PASSWORD = "qpG/mRV1N8YY"
REMOTE_TGZ_PATH = "V3C1/keyframes"

FAILED_TGZ_FILENAME = "06466.tgz"

# --- Output Config ---
OUTPUT_EMBEDDINGS_DIR = 'clip_embeddings'

# --- Config for resume ---
START_FROM_FILE = '07200.tgz'

# ---- Helper Functions (from your main script) ----
def extract_tgz(tgz_path, extract_dir):
    print(f"Extracting '{tgz_path}' to '{extract_dir}'...")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
        print(f"Successfully extracted '{tgz_path}'.")
        return True
    except Exception as e:
        print(f"Error extracting {tgz_path}: {e}")
        return False

def get_image_files(directory):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def cleanup_temp_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def load_clip_model():
    print("Loading CLIP model and processor...")
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"CLIP model loaded on device: {device}")
        return processor, model, device
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None, None

def generate_clip_embedding(image_path, processor, model, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs.pixel_values)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# ---- One-Off Processing Function ----
def process_single_tgz(tgz_name):
    print(f"\n📦 Starting reprocessing of: {tgz_name}")
    tgz_local_path = os.path.join("downloaded_tgz", tgz_name)
    extracted_path = os.path.join("temp_extracted_images", tgz_name.replace('.tgz', '_extracted'))
    embedding_dir = os.path.join(OUTPUT_EMBEDDINGS_DIR, tgz_name.replace('.tgz', ''))

    os.makedirs("downloaded_tgz", exist_ok=True)
    os.makedirs("temp_extracted_images", exist_ok=True)
    os.makedirs(OUTPUT_EMBEDDINGS_DIR, exist_ok=True)

    # 1. Download via SFTP
#    print("Connecting to SFTP to download file...")
#    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
#    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
#    sftp = paramiko.SFTPClient.from_transport(transport)
#    try:
#        remote_file = f"{REMOTE_TGZ_PATH}/{tgz_name}"
#        print(f"Downloading {tgz_name}...")
#        sftp.get(remote_file, tgz_local_path)
#        print("✅ Download complete.")
#    except Exception as e:
#        print(f"❌ Failed to download {tgz_name}: {e}")
#        return
#    finally:
#        sftp.close()
#        transport.close()

    # 2. Extract
    if not extract_tgz(tgz_local_path, extracted_path):
        cleanup_temp_dir(extracted_path)
        os.remove(tgz_local_path)
        return

    # 3. Get image files
    image_paths = get_image_files(extracted_path)
    if not image_paths:
        print(f"No images found in {tgz_name}. Skipping.")
        cleanup_temp_dir(extracted_path)
        os.remove(tgz_local_path)
        return

    # 4. Load CLIP
    processor, model, device = load_clip_model()
    if not processor or not model:
        return

    os.makedirs(embedding_dir, exist_ok=True)
    print(f"Generating embeddings for {len(image_paths)} images...")

    # 5. Generate and save embeddings
    for i, img_path in enumerate(tqdm(image_paths, desc=f"Embedding {tgz_name}")):
        rel_name = os.path.relpath(img_path, extracted_path)
        base_name = os.path.splitext(rel_name.replace("\\", "_").replace("/", "_"))[0]
        save_path = os.path.join(embedding_dir, base_name + ".npy")

        embedding = generate_clip_embedding(img_path, processor, model, device)
        if embedding is not None:
            np.save(save_path, embedding)

    print(f"✅ Saved embeddings to: {embedding_dir}")

    # 6. Clean up
    cleanup_temp_dir(extracted_path)
    os.remove(tgz_local_path)
    print("🧹 Cleanup done.")

# ---- Run It ----
if __name__ == "__main__":
    process_single_tgz(FAILED_TGZ_FILENAME)
