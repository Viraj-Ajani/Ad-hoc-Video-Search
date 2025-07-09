import os
import time
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
EMBEDDING_DIR = 'clip_embeddings'  # Path where image .npy embeddings are saved
TOP_K = 1000  # Top-N most similar images to show
QUERY_FILE = 'queries.txt'  # Input query file
OUTPUT_FILE = 'results.txt'  # Output file for results

# --- Load CLIP Model ---
def load_clip_model():
    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"CLIP model loaded on {device}")
    return processor, model, device

# --- Generate Text Embedding ---
def generate_text_embedding(query, processor, model, device):
    inputs = processor(text=query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(input_ids=inputs["input_ids"])
    return text_features.cpu().numpy().flatten()

# --- Load Image Embeddings from .npy ---
def load_all_image_embeddings(root_dir):
    embeddings = []
    image_ids = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                path = os.path.join(subdir, file)
                embedding = np.load(path)
                embeddings.append(embedding)
                relative_path = os.path.relpath(path, root_dir)
                image_ids.append(relative_path)
    return np.array(embeddings), image_ids

# --- Compute Cosine Similarity ---
def search_similar_images(text_embedding, image_embeddings, image_ids, top_k=10):
    scores = cosine_similarity([text_embedding], image_embeddings)[0]
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [image_ids[i] for i in ranked_indices]

# --- Read Queries from File ---
def read_queries(filepath):
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                query_id, query = parts
                queries.append((query_id, query))
    return queries

# --- Write Results to File ---
def write_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id, shot_ids in results:
            for shot_id in shot_ids:
                f.write(f"{query_id} {shot_id.split('/')[-1][:-4]}\n")  # strip .npy

# --- Main Logic ---
def main():
    queries = read_queries(QUERY_FILE)
    if not queries:
        print("No queries found in the input file.")
        return

    processor, model, device = load_clip_model()

    print("Loading image embeddings...")
    image_embeddings, image_ids = load_all_image_embeddings(EMBEDDING_DIR)
    if len(image_ids) == 0:
        print("No image embeddings found.")
        return

    all_results = []

    for query_id, query in queries:
        print(f"\nProcessing query ID: {query_id}")
        start_time = time.time()

        text_embedding = generate_text_embedding(query, processor, model, device)
        top_shots = search_similar_images(text_embedding, image_embeddings, image_ids, top_k=TOP_K)
        all_results.append((query_id, top_shots))

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for query '{query_id}': {elapsed:.2f} seconds")

    print(f"\nWriting results to {OUTPUT_FILE}...")
    write_results(all_results, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()

