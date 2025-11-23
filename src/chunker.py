# src/chunker.py

import os
import json
from pathlib import Path
import nltk

# Download standard Punkt tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ----------------------------
# Directories
# ----------------------------
CLEAN_DIR = Path("data/clean")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Split text into chunks
# ----------------------------
def chunk_text(text, max_words=150):
    """
    Split text into chunks of approx max_words.
    Respects sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words:
            # Finish current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_len = len(words)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ----------------------------
# Main chunking function
# ----------------------------
def main():
    all_chunks = []

    for site_folder in CLEAN_DIR.iterdir():
        if not site_folder.is_dir():
            continue

        chunk_site_folder = CHUNK_DIR / site_folder.name
        chunk_site_folder.mkdir(exist_ok=True, parents=True)

        for file_path in site_folder.iterdir():
            if not file_path.is_file() or file_path.suffix != ".txt":
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            chunk_data = []

            out_file = chunk_site_folder / (file_path.stem + "_chunks.json")
            if out_file.exists():
                out_file.unlink()  # remove old chunk file

            for i, chunk in enumerate(chunks):
                chunk_info = {
                    "id": f"{file_path.stem}_{i}",  # unique ID per chunk
                    "site": site_folder.name,
                    "source_file": file_path.name,
                    "chunk_index": i,
                    "text": chunk
                }
                chunk_data.append(chunk_info)
                all_chunks.append(chunk_info)

            # Save chunks per file as JSON
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

            print(f"Chunked {file_path} → {out_file}, total chunks: {len(chunks)}")

    # Save all chunks together for embedding
    all_chunks_file = CHUNK_DIR / "all_chunks.json"
    with open(all_chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"All chunks saved to {all_chunks_file}, total chunks: {len(all_chunks)}")

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    main()
