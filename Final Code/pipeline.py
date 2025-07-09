import os
import torch
import faiss
import glob
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from diffusers import StableDiffusionPipeline
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------- Constants --------
META_LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SD_MODEL = "stabilityai/stable-diffusion-3.5-medium"
LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"  # LLaVA model

FAISS_ROOT_DIR = "/path/to/7475_folders"
CAPTION_ROOT_DIR = "/path/to/7475_captions"  # Root for image captions (.txt)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VRAM_GB = 40

# -------- Helpers --------

def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-10)

def get_user_query() -> str:
    return input("Enter your text query: ").strip()

def generate_enhanced_queries(query: str, num_variants=10) -> List[str]:
    print("[*] Expanding query using Meta-LLaMA-3-8B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(META_LLAMA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(META_LLAMA_MODEL, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    prompt = f"Given the query: \"{query}\"\nGenerate 10 semantically similar but diverse queries, numbered 1 to 10:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    lines = decoded.split("\n")
    queries = [line[2:].strip() for line in lines if line.strip() and line[0].isdigit()]
    if len(queries) < num_variants:
        queries = [query] * num_variants
    model.cpu()
    clear_cuda_cache()
    return queries[:num_variants]

def generate_images(prompt: str, num_images=10) -> List[Image.Image]:
    print("[*] Generating images with Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL, 
                                                      torch_dtype=torch.float16, 
                                                      use_auth_token=HUGGINGFACE_API_TOKEN).to(DEVICE)
    pipe.enable_attention_slicing()
    images = [pipe(prompt, guidance_scale=7.5).images[0] for _ in range(num_images)]
    pipe.cpu()
    clear_cuda_cache()
    return images

def deduplicate_queries(queries: List[str]) -> List[str]:
    seen, unique = set(), []
    for q in queries:
        norm = q.lower().strip()
        if norm not in seen:
            seen.add(norm)
            unique.append(q)
    return unique

def deduplicate_images(images: List[Image.Image]) -> List[Image.Image]:
    seen_hashes, unique = set(), []
    for img in images:
        h = imagehash.phash(img)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(img)
    return unique

# -------- LLaVA Vision Embedder --------

from transformers import AutoTokenizer, AutoModel, LlavaNextProcessor

class LlavaEmbedder:
    def __init__(self, model_name=LLAVA_MODEL, device=DEVICE):
        print("[*] Loading LLaVA model for embeddings...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)  # Correct model class
        self.model.eval()

    def embed_text(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024))  # Adjust this size if needed based on your model config
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Text embeddings from hidden states

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 1024))  # Adjust this size if needed
        images = torch.stack([self.preprocess_image(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(images)
        return image_embeddings.cpu().numpy()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        transform = self.model.image_processor  # Use the image processor for image preparation
        return transform(image).unsqueeze(0)

    def close(self):
        self.model.cpu()
        clear_cuda_cache()

# -------- FAISS Utilities --------

def load_faiss_indexes(root_folder: str) -> List[Tuple[str, faiss.Index]]:
    print("[*] Loading FAISS indices...")
    faiss_indices = []
    folders = sorted(glob.glob(os.path.join(root_folder, "*")))
    for folder in tqdm(folders):
        for index_file in glob.glob(os.path.join(folder, "*.index")):
            try:
                idx = faiss.read_index(index_file)
                faiss_indices.append((index_file, idx))
            except Exception as e:
                print(f"[!] Skipping {index_file}: {e}")
    return faiss_indices

def search_faiss_batch(embeddings: np.ndarray, faiss_indices: List[Tuple[str, faiss.Index]], top_k=1000) -> Dict[str, float]:
    results = {}
    embeddings = embeddings.astype('float32')
    for fname, index in faiss_indices:
        if index.ntotal == 0:
            continue
        D, I = index.search(embeddings, top_k)
        for i, idx in enumerate(I):
            max_sim = np.max(D[i])
            if fname not in results or max_sim > results[fname]:
                results[fname] = max_sim
    return results

def fuse_scores(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    fused, counts = {}, {}
    for scores in score_dicts:
        for k, v in scores.items():
            fused[k] = fused.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: fused[k] / counts[k] for k in fused}

def get_top_1000_sorted(fused_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    sorted_items = sorted(fused_scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted(sorted_items[:1000], key=lambda x: x[0])

# -------- Caption Loading --------

def load_all_captions(root_dir: str) -> List[str]:
    print("[*] Loading text captions from:", root_dir)
    caption_texts = []
    for txt_file in tqdm(glob.glob(os.path.join(root_dir, "*/*.txt"))):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                caption_texts.extend([line.strip() for line in lines if line.strip()])
        except Exception as e:
            print(f"[!] Failed to read {txt_file}: {e}")
    return caption_texts

# -------- Main Pipeline --------

def main():
    query = get_user_query()
    enhanced_queries = generate_enhanced_queries(query)
    all_queries = deduplicate_queries([query] + enhanced_queries)
    images = deduplicate_images(generate_images(query))

    llava = LlavaEmbedder()

    print("[*] Embedding queries...")
    query_embeddings = normalize(llava.embed_text(all_queries))

    print("[*] Embedding images...")
    image_embeddings = normalize(llava.embed_images(images))

    print("[*] Embedding text captions from files...")
    caption_texts = load_all_captions(CAPTION_ROOT_DIR)
    caption_embeddings = normalize(llava.embed_text(caption_texts))

    llava.close()

    all_embeddings = np.vstack([query_embeddings, image_embeddings, caption_embeddings])

    faiss_indices = load_faiss_indexes(FAISS_ROOT_DIR)

    print("[*] Searching embeddings against FAISS...")
    score_dicts = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(search_faiss_batch, all_embeddings[i:i+8], faiss_indices, top_k=1000) for i in range(0, len(all_embeddings), 8)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            score_dicts.append(future.result())

    fused = fuse_scores(score_dicts)
    top_1000 = get_top_1000_sorted(fused)

    print("\nTop 1000 matched filenames and scores:")
    for fname, score in top_1000:
        print(f"{fname}\t{score:.4f}")

if __name__ == "__main__":
    main()

