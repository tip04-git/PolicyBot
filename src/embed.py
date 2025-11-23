# src/embed.py

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main(
    chunk_file_path="data/chunks/all_chunks.json",
    vector_db_dir="data/vector_db",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    CHUNK_FILE = Path(chunk_file_path)
    VECTOR_DB_DIR = Path(vector_db_dir)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load chunks
    # ----------------------------
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Each chunk must have a unique ID
    for idx, chunk in enumerate(chunks):
        if "id" not in chunk:
            chunk["id"] = idx

    # ----------------------------
    # Load existing FAISS index and metadata if available
    # ----------------------------
    index_file = VECTOR_DB_DIR / "faiss_index.bin"
    metadata_file = VECTOR_DB_DIR / "metadata.json"

    if index_file.exists() and metadata_file.exists():
        print("Loading existing FAISS index and metadata...")
        index = faiss.read_index(str(index_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        existing_ids = set(chunk["id"] for chunk in existing_metadata)
        print(f"{len(existing_ids)} chunks already indexed.")
    else:
        print("No existing index found. Creating new index...")
        index = None
        existing_metadata = []
        existing_ids = set()

    # ----------------------------
    # Filter out already indexed chunks
    # ----------------------------
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        print("No new chunks to index.")
        return

    texts = [c["text"] for c in new_chunks]

    # ----------------------------
    # Generate embeddings
    # ----------------------------
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print(f"Generating embeddings for {len(texts)} new chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # ----------------------------
    # Build or update FAISS index
    # ----------------------------
    dim = embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)  # L2 distance
    else:
        if index.d != dim:
            raise ValueError(f"Dimension mismatch! Existing index dim={index.d}, new embeddings dim={dim}")

    index.add(embeddings)
    print(f"FAISS index now has {index.ntotal} vectors.")

    # ----------------------------
    # Update metadata
    # ----------------------------
    all_metadata = existing_metadata + new_chunks

    # Save index and metadata
    faiss.write_index(index, str(index_file))
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Vector DB saved in {VECTOR_DB_DIR}, total chunks indexed: {len(all_metadata)}")


# ----------------------------
# Allow running as script
# ----------------------------
if __name__ == "__main__":
    main()
