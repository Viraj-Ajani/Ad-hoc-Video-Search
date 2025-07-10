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
from codecarbon import EmissionsTracker

# === Configuration ===
INPUT_TGZ_DIR = "path/to/tgz/files"
OUTPUT_DIR = "./vl2_captions"
TEMP_DIR = "./temp_extracted"
CHECKPOINT_FILE = "./checkpoint.json"
MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"
PROMPT = "Describe this image in detail:"
BASE_BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# === Dynamic Batch Size Based on VRAM ===
def adjust_batch_size(base_batch=4):
    """Adjust batch size based on available VRAM in GB"""
    free_mem = torch.cuda.mem_get_info()[0] / 1e9  # bytes to GB
    if free_mem >= 20:
        return base_batch * 4
    elif free_mem >= 15:
        return base_batch * 3
    elif free_mem >= 10:
        return base_batch * 2
    else:
        return base_batch

BATCH_SIZE = adjust_batch_size(BASE_BATCH_SIZE)

# === Setup Directories ===
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(TEMP_DIR).mkdir(exist_ok=True)

# === Load Checkpoint ===
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        processed = set(json.load(f))
else:
    processed = set()

# === Load Model ===
try:
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    print(f"✅ Loaded {MODEL_ID} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

def process_batch(image_paths, tgz_name):
    """Process images and generate captions"""
    output_folder = Path(OUTPUT_DIR) / tgz_name
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

def process_tgz(tgz_path):
    """Process .tgz files with atomic checkpointing"""
    tgz_name = Path(tgz_path).stem
    extracted_path = Path(TEMP_DIR) / tgz_name

    try:
        if str(tgz_path) in processed:
            return

        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=extracted_path)

        image_paths = [
            str(p) for p in extracted_path.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and str(p) not in processed
        ]

        for i in range(0, len(image_paths), BATCH_SIZE):
            process_batch(image_paths[i:i+BATCH_SIZE], tgz_name)

    except Exception as e:
        print(f"❌ {tgz_path} failed: {e}")
    finally:
        shutil.rmtree(extracted_path, ignore_errors=True)
        temp_ckpt = f"{CHECKPOINT_FILE}.tmp"
        with open(temp_ckpt, "w") as f:
            json.dump(list(processed), f)
        os.replace(temp_ckpt, CHECKPOINT_FILE)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(f"⚙️ Device: {DEVICE} | Model: {MODEL_ID} | Batch: {BATCH_SIZE}")

    # Start carbon tracking
    tracker = EmissionsTracker(project_name="DeepSeek-VL2-Captioning", output_dir=".", output_file="carbon_footprint.csv")
    tracker.start()

    tgz_files = sorted(Path(INPUT_TGZ_DIR).glob("*.tgz"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_tgz, tgz_files)

    emissions = tracker.stop()
    print(f"✅ Caption generation complete!")
    print(f"🌱 Carbon footprint: {emissions:.6f} kg CO2eq")

