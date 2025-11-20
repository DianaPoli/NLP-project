# fertility_experiment.py

import csv
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


# 1. CONFIGURATION
SEED = 42
NUM_SENTENCES = 1000   #number of sentences to sample
DATA_PATH = "data/italian_corpus.txt"  #path to the Italian text corpus

#Tokenizers to compare
TOKENIZERS = {
    "LLaMA3.1-Base": "meta-llama/Llama-3.1-8B",
    "LAPT": "SemanticAlignment/Llama-3.1-8B-Italian-LAPT",
    "SAVA": "SemanticAlignment/Llama-3.1-8B-Italian-SAVA",
    "FVT":  "SemanticAlignment/Llama-3-1-8B-Italian-FVT"
}

OUTPUT_CSV = "results/fertility_results.csv"
OUTPUT_PLOT = "results/fertility_plot.png"

# 2. FUNCTIONS
def load_sentences(path, num):
    #Load N Italian sentences from a .txt file
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip().split()) > 3]
    return random.sample(lines, num)


def token_fertility(tokenizer, text):
    #Token fertility = num_token / num_words
    return len(tokenizer(text).input_ids) / len(text.split())


def compute_fertility_for_tokenizer(name, tokenizer_id, sentences):
    print(f"\nCalculating fertility for: {name} ({tokenizer_id})")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    fertilities = []
    for sent in sentences:
        fertilities.append(token_fertility(tokenizer, sent))

    avg = sum(fertilities) / len(fertilities)
    print(f"Average fertility {name}: {avg:.3f}")
    return avg


# 3. MAIN
def main():
    print("Loading Italian sentences…")
    sentences = load_sentences(DATA_PATH, NUM_SENTENCES)
    print(f"{len(sentences)} sentences loaded.")

    results = {}

    # Compute fertility for each tokenizer
    for name, path in TOKENIZERS.items():
        results[name] = compute_fertility_for_tokenizer(name, path, sentences)

    # 4. Save CSV
    import os
    os.makedirs("results", exist_ok=True)
    print("\nSaving results to CSV")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokenizer", "Fertility"])
        for name, avg in results.items():
            writer.writerow([name, avg])

    print(f"CSV saved to: {OUTPUT_CSV}")

    # 5. Plot
    print("Generating plot")
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color=["gray", "green", "blue", "orange"])
    plt.title("Token Fertility on adapted tokenizers")
    plt.ylabel("Fertility (tokens per word)")
    plt.savefig(OUTPUT_PLOT)
    plt.show()

    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
