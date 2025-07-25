import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from codecarbon import EmissionsTracker
import logging
from codecarbon.output import LoggerOutput

# ---- Paths ---- #
root_image_dir = "./root_images"        # Root directory with subfolders
output_dir = "./captions"               # Where captions will be saved
processed_log_path = "processed_files_v3c2.txt"
os.makedirs(output_dir, exist_ok=True)

# ---- Setup Carbon Emissions Tracker ---- #
logger = logging.getLogger("carbon_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger_output = LoggerOutput(logger=logger)

tracker = EmissionsTracker(
    project_name="v3c2",
    output_file="emissions_v3c2.csv",
    logging_logger=logger_output
)
tracker.start()

# ---- Load Model and Processor ---- #
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# ---- Load Already Processed ---- #
if os.path.exists(processed_log_path):
    with open(processed_log_path, "r") as f:
        processed_files = set(line.strip() for line in f)
else:
    processed_files = set()

# ---- Process All Images in Subfolders ---- #
supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
for dirpath, _, filenames in os.walk(root_image_dir):
    for filename in sorted(filenames):
        if not filename.lower().endswith(supported_exts):
            continue

        full_image_path = os.path.join(dirpath, filename)
        base_filename = os.path.splitext(filename)[0]
        txt_output_path = os.path.join(output_dir, f"{base_filename}.txt")

        if base_filename in processed_files:
            continue

        print(f"🖼️ Captioning: {full_image_path}")

        try:
            # Create conversation format
            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\n<|ref|>Describe the image.<|/ref|>.",
                    "images": [full_image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

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

            # Save caption
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(answer.strip())

            # Update log
            with open(processed_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"{base_filename}\n")
            processed_files.add(base_filename)

        except Exception as e:
            print(f"❌ Failed to caption {filename}: {e}")

# ---- Final Emissions ---- #
emissions = tracker.stop()
print(f"\n🌍 Total CO₂ emissions: {emissions:.6f} kg")
print("📄 Emissions log saved to: emissions_v3c2.csv")
