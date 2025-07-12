import os
import json
import atexit
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from concurrent.futures import ThreadPoolExecutor
from codecarbon import EmissionsTracker

# === Configuration ===
INPUT_FOLDER_DIR = "/media/irlab/890bd749-cf2c-4f59-95fa-25153d050c89/Viraj/AVS/Dataset"
OUTPUT_DIR = "./vl2_captions"
CHECKPOINT_FILE = "./checkpoint.json"
MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"
PROMPT = "<|User|>\n<image>\n"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
processed = set(json.load(open(CHECKPOINT_FILE)) if Path(CHECKPOINT_FILE).exists() else [])

# Load model & processor
processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

# Track emissions for the whole run
tracker = EmissionsTracker(project_name="vl2_captioning", measure_power_secs=60)
tracker.start()

def save_checkpoint():
    tmp = CHECKPOINT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(list(processed), f)
    os.replace(tmp, CHECKPOINT_FILE)

def shutdown_handler():
    emissions = tracker.stop()
    print(f"\n📊 Total CO₂ emissions this session: {emissions:.4f} kg")
atexit.register(shutdown_handler)

def process_batch(image_paths, folder_name):
    out_dir = Path(OUTPUT_DIR) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Prepare conversations
        conversations = []
        for img in image_paths:
            conversations += [
                {"role": "<|User|>", "content": PROMPT, "images": [img]},
                {"role": "<|Assistant|>", "content": ""}
            ]
        pil_images = load_pil_images(conversations)
        inputs = processor(conversations=conversations, images=pil_images, force_batchify=True, system_prompt="").to(DEVICE)

        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        # Write captions
        idx = 0
        for img_path in image_paths:
            cap = processor.tokenizer.decode(outputs[idx].cpu(), skip_special_tokens=True).strip()
            idx += 1
            stem = Path(img_path).stem
            with open(out_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.write(cap)
            processed.add(str(img_path))

        save_checkpoint()

    except Exception as e:
        print(f"⚠️ Batch error on {folder_name}: {e}")
        save_checkpoint()

def process_folder(folder_path):
    folder = Path(folder_path).name
    img_list = [
        str(p) for p in Path(folder_path).rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and str(p) not in processed
    ]
    for i in range(0, len(img_list), BATCH_SIZE):
        process_batch(img_list[i:i+BATCH_SIZE], folder)

if __name__ == "__main__":
    print(f"Device: {DEVICE}, Model: {MODEL_ID}, Batch size: {BATCH_SIZE}")
    folders = [p for p in Path(INPUT_FOLDER_DIR).iterdir() if p.is_dir()]
    with ThreadPoolExecutor(max_workers=4) as exe:
        exe.map(process_folder, folders)
    # Emissions printed on exit via atexit
    print("✅ Caption generation complete!")

