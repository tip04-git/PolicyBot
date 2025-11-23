# src/full_pipeline.py

import os
import yaml
import logging
from crawler import main as crawl
from downloader import main as download
from preprocess import main as clean
from chunker import main as chunk
from embed import main as embed

# ----------------------------
# Setup logging
# ----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------
# Track already downloaded / processed files
# ----------------------------
ALREADY_DOWNLOADED_PATH = "data/raw/already_downloaded.yaml"
ALREADY_EMBEDDED_PATH = "data/vector_db/already_embedded.yaml"

def load_yaml(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

def run_pipeline():
    logging.info("Starting full pipeline...")

    # ----------------------------
    # Crawl
    # ----------------------------
    try:
        crawl()
        logging.info("Crawling completed successfully.")
    except Exception as e:
        logging.error(f"Crawling failed: {e}")

    # ----------------------------
    # Download
    # ----------------------------
    already_downloaded = load_yaml(ALREADY_DOWNLOADED_PATH)
    try:
        download()  # downloader.py already handles skipping existing files
        logging.info("Downloading completed successfully.")
    except Exception as e:
        logging.error(f"Downloading failed: {e}")
    finally:
        # update already_downloaded.yaml
        save_yaml(already_downloaded, ALREADY_DOWNLOADED_PATH)

    # ----------------------------
    # Preprocess
    # ----------------------------
    try:
        clean()  # skips non-PDF files, outputs cleaned text
        logging.info("Preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")

    # ----------------------------
    # Chunk
    # ----------------------------
    try:
        chunk()  # generates chunks per file and all_chunks.json
        logging.info("Chunking completed successfully.")
    except Exception as e:
        logging.error(f"Chunking failed: {e}")

    # ----------------------------
    # Embed
    # ----------------------------
    already_embedded = load_yaml(ALREADY_EMBEDDED_PATH)
    try:
        embed()  # embed.py can be modified to skip already embedded chunks
        logging.info("Embedding completed successfully.")
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
    finally:
        save_yaml(already_embedded, ALREADY_EMBEDDED_PATH)

    logging.info("Pipeline completed!")

if __name__ == "__main__":
    run_pipeline()
