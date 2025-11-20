# inference_time_light.py

import random
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. CONFIGURATION
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NUM_SENTENCES = 50
DATA_PATH = "data/italian_corpus.txt"

MODELS = {
    "LLaMA3.1-Base": "meta-llama/Llama-3.1-8B",
    "LAPT": "SemanticAlignment/Llama-3.1-8B-Italian-LAPT",
    "SAVA": "SemanticAlignment/Llama-3.1-8B-Italian-SAVA",
    "FVT":  "SemanticAlignment/Llama-3-1-8B-Italian-FVT"
}

OUTPUT_CSV = "results/inference_time_light_results.csv"
PLOT_INFERENCE = "results/inference_time_light_plot.png"
MAX_NEW_TOKENS = 5  #number of tokens to generate during inference

# 2. FUNCTIONS
def load_sentences(path, num):
    """Load N Italian sentences from a .txt file"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip().split()) > 3]
    return random.sample(lines, num)

def measure_inference_time_light(tokenizer, model, sentences):
    """Measure average time to tokenize + generate few tokens"""
    times = []
    for s in sentences:
        start = time.perf_counter()

        # Tokenization
        inputs = tokenizer(s, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generation
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)

def free_gpu(model):
    """Free GPU memory after a model is used"""
    if model is not None:
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

# 3. MAIN
def main():
    print("Loading sentences…")
    sentences = load_sentences(DATA_PATH, NUM_SENTENCES)
    print(f"{len(sentences)} sentences loaded.")

    import os
    os.makedirs("results", exist_ok=True)

    results = {}

    for name, path in MODELS.items():
        print(f"\nProcessing model: {name} ({path})")

        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
            model.eval()
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            results[name] = None
            continue

        try:
            avg_time = measure_inference_time_light(tokenizer, model, sentences)
            print(f"Average light inference time for {name}: {avg_time:.6f} sec per sentence")
            results[name] = avg_time
        except RuntimeError as e:
            print(f"Failed light inference for {name}: {e}")
            results[name] = None

        free_gpu(model)

    # Save CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "AvgLightInferenceTimeSec"])
        for name, t in results.items():
            writer.writerow([name, t if t else "N/A"])
    print(f"CSV saved at {OUTPUT_CSV}")

    # Plot
    names = list(results.keys())
    times = [results[n] if results[n] else 0 for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, times, color=["gray", "green", "blue", "orange"])
    plt.title(f"Light Inference Time ({NUM_SENTENCES} sentences, {MAX_NEW_TOKENS} tokens)")
    plt.ylabel("Seconds per sentence")
    plt.savefig(PLOT_INFERENCE)
    plt.show()
    print(f"Light inference plot saved at {PLOT_INFERENCE}")

if __name__ == "__main__":
    main()
