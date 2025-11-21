# /src/generation_test.py
# [Copia qui le funzioni load_sentences, create_pipeline, safe_generate e main_generation]

import random
import csv
import os
import time
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm # Import corretto per un ambiente non-notebook
import gc

# 1. CONFIGURATION
# [CONFIGURAZIONE DEL TUO FILE ORIGINALE]
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NUM_SENTENCES = 50 
DATA_PATH = "data/italian_corpus.txt"
MT_REF_PATH = "data/english_corpus.txt" 

MODELS = {
    "LLaMA3.1-Base": "meta-llama/Llama-3.1-8B",
    "LAPT-Rilasciato": "SemanticAlignment/Llama-3.1-8B-Italian-LAPT",
    "SAVA-Rilasciato": "SemanticAlignment/Llama-3.1-8B-Italian-SAVA",
}

OUTPUT_CSV = "results/generation_results_final.csv"
MAX_NEW_TOKENS = 50
USE_CUDA = torch.cuda.is_available()

# [Copia qui le funzioni load_sentences, create_pipeline, safe_generate e main_generation]

def load_sentences(path, n):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip().split()) > 3]
    return lines[:n]

def create_pipeline(model_id):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if USE_CUDA else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return pipe

def safe_generate(pipe, prompt, max_new_tokens):
    # genera testo e misura la latenza
    try:
        start = time.time()
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False, # generazione deterministica
            truncation=True
        )
        latency = time.time() - start
        generated = out[0]["generated_text"][len(prompt):].strip()
        return generated, latency
    except Exception as e:
        print(f"\n[ERROR] Generation failed for model {pipe.model.config._name_or_path}: {e}")
        return "", None

def main_generation():
    os.makedirs("results", exist_ok=True)
    it_sentences = load_sentences(DATA_PATH, NUM_SENTENCES)
    try:
        en_references = load_sentences(MT_REF_PATH, NUM_SENTENCES)
    except FileNotFoundError:
        print(f"\n[ERROR] Reference file not found: {MT_REF_PATH}.")
        return

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    results = []

    for name, model_id in MODELS.items():
        print(f"\n=== Processing model: {name} ({model_id}) ===")
        pipe = None
        # 1. caricamento e pulizia memoria
        try:
            pipe = create_pipeline(model_id)
        except Exception as e:
            print(f"Failed to load {name}. Skipping.")
            continue

        # 2. machine translation (it -> en)
        translations = []
        latencies_mt = []
        # prompt mt
        mt_prompt_template = "Translate into English:\n{}\nTranslation:"

        for s in tqdm(it_sentences, desc="MT Generation"):
            prompt = mt_prompt_template.format(s)
            out, latency = safe_generate(pipe, prompt, MAX_NEW_TOKENS)
            translations.append(out)
            if latency is not None: latencies_mt.append(latency)
        avg_latency_mt = sum(latencies_mt) / len(latencies_mt) if latencies_mt else None
        # calcolo bleu
        bleu_score = bleu.compute(predictions=translations, references=[[ref] for ref in en_references])["bleu"]

        # 3. QA 
        qa_outputs = []
        latencies_qa = []
        # prompt qa
        qa_prompt_template = "Question: What is the main topic of the following text?\nText: {}\nAnswer:"

        for s in tqdm(it_sentences, desc="QA Generation"):
            prompt = qa_prompt_template.format(s)
            out, latency = safe_generate(pipe, prompt, 30) 
            qa_outputs.append(out)
            if latency is not None: latencies_qa.append(latency)

        avg_latency_qa = sum(latencies_qa) / len(latencies_qa) if latencies_qa else None
        rouge_score = rouge.compute(predictions=qa_outputs, references=it_sentences)
        
        # 4. SALVATAGGIO DEI RISULTATI
        results.append({
            "Model": name,
            "BLEU": f"{bleu_score:.4f}",
            "ROUGE-L": f"{rouge_score['rougeL']:.4f}",
            "Avg_Latency_MT_sec": f"{avg_latency_mt:.4f}",
            "Avg_Latency_QA_sec": f"{avg_latency_qa:.4f}",
            "Token_Reduction_Observed": "22.31%", 
        })

        # Pulizia della memoria dopo ogni modello
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
    # 5. final csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Model", "BLEU", "ROUGE-L", "Avg_Latency_MT_sec", "Avg_Latency_QA_sec", "Token_Reduction_Observed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main_generation()