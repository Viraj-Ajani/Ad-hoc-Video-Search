import os
import tarfile
import tempfile
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging


# ---- Settings ---- #
model_path = "deepseek-ai/deepseek-vl2-tiny"
tgz_dir = "./tgz_path"
output_dir = "./captions"
processed_log_path = "processed_files_v3c1.txt"
os.makedirs(output_dir, exist_ok=True)

# ---- Load Tracker ---- #
logger = logging.getLogger("carbon_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())  # Print to console

logger_output = LoggerOutput(logger=logger)
tracker = EmissionsTracker(project_name="v3c1", output_file="emissions_v3c1.csv", logging_logger=logger_output)
tracker.start()

# ---- Load Model and Processor ---- #
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# ---- Load Processed Files ---- #
if os.path.exists(processed_log_path):
    with open(processed_log_path, "r") as f:
        processed_files = set(line.strip() for line in f)
else:
    processed_files = set()

# ---- Processing Begins ---- #
for tgz_file in sorted(os.listdir(tgz_dir)):
    if not tgz_file.endswith(".tgz"):
        continue

    tgz_path = os.path.join(tgz_dir, tgz_file)
    print(f"📦 Processing archive: {tgz_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        for root, _, files in os.walk(temp_dir):
            for filename in sorted(files):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                base_filename = os.path.splitext(filename)[0]
                txt_output_path = os.path.join(output_dir, f"{base_filename}.txt")

                # Skip if already processed
                if base_filename in processed_files:
                    continue

                image_path = os.path.join(root, filename)
                print(f"🖼️ Captioning image: {filename}")

                try:
                    # Construct conversation
                    conversation = [
                        {
                            "role": "<|User|>",
                            "content": "<image>\n<|ref|>Describe the image.<|/ref|>.",
                            "images": [image_path],
                        },
                        {"role": "<|Assistant|>", "content": ""},
                    ]

                    # Load and process image
                    pil_images = load_pil_images(conversation)
                    prepare_inputs = vl_chat_processor(
                        conversations=conversation,
                        images=pil_images,
                        force_batchify=True,
                        system_prompt=""
                    ).to(vl_gpt.device)

                    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                    # Generate caption
                    with torch.no_grad():
                        outputs = vl_gpt.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=prepare_inputs.attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=512,
                            do_sample=False,
                            use_cache=True
                        )

                    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

                    # Save output
                    with open(txt_output_path, "w", encoding="utf-8") as out_f:
                        out_f.write(answer.strip())

                    # Log processed file
                    with open(processed_log_path, "a", encoding="utf-8") as log_f:
                        log_f.write(f"{base_filename}\n")
                    processed_files.add(base_filename)

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

# ---- Final Emissions Report ---- #
emissions = tracker.stop()
print(f"\n🌍 Total CO₂ emissions: {emissions:.6f} kg")
print("📄 Emission log saved to: emissions_v3c1.csv")

