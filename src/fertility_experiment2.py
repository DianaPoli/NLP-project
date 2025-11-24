# fertility_experiment.py (FINAL ADAPTATION)

import csv
import random
import matplotlib.pyplot as plt # Importato all'inizio
from transformers import AutoTokenizer
import os # Importato all'inizio
import torch

# 1. CONFIGURATION
SEED = 42
NUM_SENTENCES = 1000 
DATA_PATH = "data/italian_corpus.txt"
random.seed(SEED)

# Definisci il percorso del tuo modello addestrato in Mini-CT
MY_SAVA_ADAPTER_DIR = "./results/mct_sava_output"

# Tokenizers to compare (aggiungiamo il tuo modello riprodotto)
TOKENIZERS = {
    "LLaMA3.1-Base": "meta-llama/Llama-3.1-8B",
    "LAPT-HF": "SemanticAlignment/Llama-3.1-8B-Italian-LAPT",
    "SAVA-HF": "SemanticAlignment/Llama-3.1-8B-Italian-SAVA",
    "FVT-HF": "SemanticAlignment/Llama-3-1-8B-Italian-FVT",
    "SAVA-MCT-REPRODUCED": MY_SAVA_ADAPTER_DIR, 
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
    num_words = len(text.split())
    if num_words == 0:
        return 0.0
    return len(tokenizer(text).input_ids) / num_words


def compute_fertility_for_tokenizer(name, tokenizer_id, sentences):
    print(f"\nCalculating fertility for: {name} ({tokenizer_id})")
    
    # CRITICO: Usa trust_remote_code=True per il tokenizer Minerva e l'adapter locale
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

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
        # Verifico se l'adapter esiste prima di tentare di caricarlo
        if name == "SAVA-MCT-REPRODUCED" and not os.path.exists(path):
            print(f"Skipping {name}: Local adapter directory not found at {path}.")
            continue
            
        results[name] = compute_fertility_for_tokenizer(name, path, sentences)

    # 4. Aggiungi l'analisi dell'efficienza relativa nel CSV
    mistral_fertility = results.get("LLaMA3.1-Base")
    reproduced_fertility = results.get("SAVA-MCT-REPRODUCED")
    
    if mistral_fertility is not None and reproduced_fertility is not None:
        reduction = (mistral_fertility - reproduced_fertility) / mistral_fertility
        results["Token_Reduction_%"] = reduction * 100
        print(f"\nToken Fertility Reduction (Reproduced Model): {reduction:.2%}")


    # 5. Save CSV
    os.makedirs("results", exist_ok=True)
    print("\nSaving results to CSV")
    
    # Include la riga di riduzione nel CSV
    fieldnames = ["Tokenizer", "Fertility"]
    if "Token_Reduction_%" in results:
        fieldnames.append("Token_Reduction_%")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        
        # Scrivi le righe dei modelli
        for name, avg in results.items():
            if name not in ["Token_Reduction_%"]:
                writer.writerow([name, avg])
        
        # Scrivi la riga del riassunto riduzione
        if "Token_Reduction_%" in results:
            writer.writerow(["Reduction_vs_Base", results["Token_Reduction_%"]])


    print(f"CSV saved to: {OUTPUT_CSV}")

    # 6. Plot
    print("Generating plot")
    names_to_plot = [name for name in results.keys() if name not in ["Token_Reduction_%"]]
    values_to_plot = [results[name] for name in names_to_plot]

    plt.figure(figsize=(12, 6))
    plt.bar(names_to_plot, values_to_plot, color=["gray", "green", "blue", "orange", "purple"])
    plt.title("Token Fertility Comparison (LLaMA Base vs Adapted vs Reproduced)")
    plt.ylabel("Fertility (tokens per word)")
    plt.savefig(OUTPUT_PLOT)
    plt.show()

    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()