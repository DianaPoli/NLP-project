# @title 5.1 Esecuzione di `token_count_and_speed.py` (Efficienza di Tokenizzazione)

import csv
import random
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os
import torch

# 1. CONFIGURATION
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NUM_SENTENCES = 500  # Numero di frasi da campionare per il test
DATA_PATH = "data/italian_corpus.txt"

# Tokenizers da confrontare: Sorgente (inglese) vs Target (italiano)
TOKENIZERS = {
    "Mistral-Base": "mistralai/Mistral-7B-v0.1", 
    "Minerva-Target": "sapienzanlp/Minerva-3B-base-v1.0",
}

OUTPUT_CSV = "results/token_count_and_speed.csv"
PLOT_COUNT = "results/token_count_plot.png"
PLOT_SPEED = "results/tokenization_speed_plot.png"


# 2. FUNCTIONS
def load_sentences(path, num):
    # Load N sentences from file.
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip().split()) > 3]
    return random.sample(lines, min(num, len(lines)))

def count_tokens(tokenizer, text):
    # Return number of tokens for a sentence.
    return len(tokenizer(text).input_ids)

def measure_tokenization_time(tokenizer, sentences):
    # Measure average time to tokenize a sentence.
    times = []
    # Usiamo un warm-up per stabilizzare i tempi
    for _ in range(5):
        tokenizer(sentences[0])
        
    for s in sentences:
        start = time.perf_counter()
        tokenizer(s)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)

# 3. MAIN
def main_token_speed():
    print("Loading sentences")
    sentences = load_sentences(DATA_PATH, NUM_SENTENCES)
    print(f"{len(sentences)} sentences loaded.")

    results = {}

    for name, path in TOKENIZERS.items():
        print(f"\nProcessing tokenizer: {name} ({path})")
        
        # Caricamento del tokenizer con trust_remote_code per Minerva
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        # Token count average
        token_counts = [count_tokens(tok, s) for s in sentences]
        avg_count = sum(token_counts) / len(token_counts)

        # Inference time average
        avg_time = measure_tokenization_time(tok, sentences)

        print(f"Avg token count: {avg_count:.2f}")
        print(f"Avg tokenization time: {avg_time:.6f} sec")

        results[name] = {
            "count": avg_count,
            "time": avg_time
        }

    # Calcolo del Guadagno di Efficienza
    count_mistral = results.get("Mistral-Base", {}).get("count", 0)
    count_minerva = results.get("Minerva-Target", {}).get("count", 0)
    time_mistral = results.get("Mistral-Base", {}).get("time", 0)
    time_minerva = results.get("Minerva-Target", {}).get("time", 0)
    
    if count_mistral > 0 and count_minerva > 0:
        reduction_token = (count_mistral - count_minerva) / count_mistral
        print(f"\nToken Reduction %: {reduction_token:.2%}")
    if time_mistral > 0 and time_minerva > 0:
        speed_increase = (time_mistral - time_minerva) / time_mistral
        print(f"Tokenization Speed Increase %: {speed_increase:.2%}")


    # 4 Save CSV
    os.makedirs("results", exist_ok=True)
    print("\nSaving CSV")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokenizer", "AvgTokenCount", "AvgTokenizationTime"])
        for name, res in results.items():
            writer.writerow([name, res["count"], res["time"]])

    # Plot token count
    names = list(results.keys())
    counts = [results[n]["count"] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, counts, color=["gray", "teal"])
    plt.title("Average Token Count Comparison")
    plt.ylabel("Average number of tokens")
    plt.savefig(PLOT_COUNT)
    plt.show()

    # 5 Plot tokenization speed
    times = [results[n]["time"] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, times, color=["gray", "teal"])
    plt.title("Average Tokenization Speed")
    plt.ylabel("Seconds per sentence")
    plt.savefig(PLOT_SPEED)
    plt.show()

    print("Plots saved.")

if __name__ == "__main__":
    main_token_speed()