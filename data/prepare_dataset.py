# prepare_dataset.py

import os
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.it"
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "italian_corpus.txt"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

NUM_SENTENCES_TO_SAVE = 2000
MIN_WORD_COUNT = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="train",
    streaming=True)

sentences_saved = 0

with tqdm(total=NUM_SENTENCES_TO_SAVE, desc="Frasi salvate") as pbar:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        for sample in dataset:
            if sentences_saved >= NUM_SENTENCES_TO_SAVE:
                break

            article_text = sample['text']

            for line in article_text.split('\n'):
                text = line.strip()

                if text and not text.startswith("=") and len(text.split()) > MIN_WORD_COUNT:
                    f.write(text + "\n")
                    sentences_saved += 1
                    pbar.update(1)

                    if sentences_saved >= NUM_SENTENCES_TO_SAVE:
                        break