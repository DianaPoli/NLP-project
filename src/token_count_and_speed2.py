# token_count_and_speed.py

import csv
import random
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os # Necessario per os.path.exists
import torch

# 1 CONFIGURATION
SEED = 42
NUM_SENTENCES = 500  # less then 1000
DATA_PATH = "data/italian_corpus.txt"
random.seed(SEED)

# Definisci il percorso del tuo modello addestrato in Mini-CT
MY_SAVA_ADAPTER_DIR = "./results/mct_sava_output"

# Tokenizers to compare
TOKENIZERS = {
    "LLaMA3.1-Base": "meta-llama/Llama-3.1-8B",
    "LAPT-HF": "SemanticAlignment/Llama-3.1-8B-Italian-LAPT",
    "SAVA-HF": "SemanticAlignment/Llama-3.1-8B-Italian-SAVA",
    "FVT-HF": "SemanticAlignment/Llama-3-1-8B-Italian-FVT",
    # INSERIMENTO DEL MODELLO TRAINATO DA TE
    "SAVA-MCT-REPRODUCED": MY_SAVA_ADAPTER_DIR, 
}

OUTPUT_CSV = "results/token_count_and_speed.csv"
PLOT_COUNT = "results/token_count_plot.png"
PLOT_SPEED = "results/tokenization_speed_plot.png"


# 2 FUNCTIONS
def load_sentences(path, num):
    #Load N sentences from file.
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip().split()) > 3]
    # Assicurati di non campionare più del numero totale di frasi disponibili
    return random.sample(lines, min(num, len(lines)))


def count_tokens(tokenizer, text):
    #Return number of tokens for a sentence.
    return len(tokenizer(text).input_ids)

def measure_tokenization_time(tokenizer, sentences):
    #Measure average time to tokenize a sentence.
    times = []
    # Aggiungi warm-up per misurazioni più stabili
    if sentences:
        for _ in range(5):
            tokenizer(sentences[0])
            
    for s in sentences:
        start = time.perf_counter()
        tokenizer(s)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)

# 3 MAIN
def main():
    print("Loading sentences")
    sentences = load_sentences(DATA_PATH, NUM_SENTENCES)
    print(f"{len(sentences)} sentences loaded.")

    results = {}

    for name, path in TOKENIZERS.items():
        # --- LOGICA DI SALTO PER IL MODELLO NON TROVATO ---
        if name == "SAVA-MCT-REPRODUCED" and not os.path.isdir(path):
            print(f"\nSkipping {name}: Local training directory not found at {path}. (Run sava_mini_train.py first)")
            continue
            
        print(f"\nProcessing tokenizer: {name} ({path})")
        
        # CRITICO: Usa trust_remote_code=True per Minerva e l'adapter locale
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True) 

        # token count average
        token_counts = [count_tokens(tok, s) for s in sentences]
        avg_count = sum(token_counts) / len(token_counts)

        # inference time average
        avg_time = measure_tokenization_time(tok, sentences)

        print(f"Avg token count: {avg_count:.2f}")
        print(f"Avg tokenization time: {avg_time:.6f} sec")

        results[name] = {
            "count": avg_count,
            "time": avg_time
        }

    # 4  Save CSV
    os.makedirs("results", exist_ok=True)
    print("\nSaving CSV")
    
    # Calcola e aggiungi la riga di riduzione per il tuo modello riprodotto
    mistral_count = results.get("LLaMA3.1-Base", {}).get("count")
    reproduced_count = results.get("SAVA-MCT-REPRODUCED", {}).get("count")
    
    if mistral_count is not None and reproduced_count is not None:
        reduction = (mistral_count - reproduced_count) / mistral_count
        print(f"Total Token Reduction (Reproduced Model): {reduction:.2%}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokenizer", "AvgTokenCount", "AvgTokenizationTime"])
        for name, res in results.items():
            writer.writerow([name, res["count"], res["time"]])

    # Plot token count
    names = list(results.keys())
    counts = [results[n]["count"] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, counts, color=["gray", "green", "blue", "orange", "purple"])
    plt.title("Average Token Count Comparison")
    plt.ylabel("Average number of tokens")
    plt.savefig(PLOT_COUNT)
    plt.show()

    # 5 Plot tokenization speed
    times = [results[n]["time"] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, times, color=["gray", "green", "blue", "orange", "purple"])
    plt.title("Average Tokenization Speed")
    plt.ylabel("Seconds per sentence")
    plt.savefig(PLOT_SPEED)
    plt.show()

    print("Plots saved.")

if __name__ == "__main__":
    main()