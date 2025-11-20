import random
import csv
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate

#1. CONFIGURATION
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NUM_SENTENCES = 50
DATA_PATH = "data/italian_corpus.txt"
MT_REF_PATH = "data/english_corpus.txt"

MODELS = {
    "LLaMA3.1-Base": "meta-llama/Llama-3.1-8B",
    "LAPT": "SemanticAlignment/Llama-3.1-8B-Italian-LAPT",
    "SAVA": "SemanticAlignment/Llama-3.1-8B-Italian-SAVA",
    "FVT":  "SemanticAlignment/Llama-3-1-8B-Italian-FVT"
}

OUTPUT_CSV = "results/generation_results.csv"
MAX_NEW_TOKENS = 50
NUM_SAMPLES_TO_SAVE = 5

USE_CUDA = torch.cuda.is_available()


#2. FUNCTIONS
def load_sentences(path, n):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip().split()) > 3]
    return lines[:n]


def create_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if USE_CUDA else torch.float32,
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    if USE_CUDA:
        pipe.model.to("cuda")

    return pipe


def safe_generate(pipe, prompt, max_new_tokens=MAX_NEW_TOKENS):
    try:
        start = time.time()
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            truncation=True
        )
        latency = time.time() - start
        generated = out[0]["generated_text"][len(prompt):].strip()
        return generated, latency

    except Exception as e:
        print(f"Generation failed: {e}")
        return "", None


#3. MAIN 
def main():
    os.makedirs("results", exist_ok=True)

    it_sentences = load_sentences(DATA_PATH, NUM_SENTENCES)
    en_references = load_sentences(MT_REF_PATH, NUM_SENTENCES)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    results = []

    for name, model_id in MODELS.items():
        print(f"\n=== Processing model: {name} ===")

        try:
            pipe = create_pipeline(model_id)
        except Exception as e:
            print(f"Failed to load {name}: {e}")

            results.append({
                "Tokenizer": name,
                "BLEU": None,
                "ROUGE-1": None,
                "ROUGE-2": None,
                "ROUGE-L": None,
                "Avg_Latency_MT": None,
                "Avg_Latency_QA": None,
                "MT_Sample": "",
                "QA_Sample": ""
            })
            continue

        #MACHINE TRANSLATION
        translations = []
        latencies_mt = []

        for s in it_sentences:
            prompt = f"Translate into English:\n{s}\nTranslation:"
            out, latency = safe_generate(pipe, prompt)
            translations.append(out)
            if latency is not None:
                latencies_mt.append(latency)

        avg_latency_mt = sum(latencies_mt) / len(latencies_mt) if latencies_mt else None

        # BLEU expects predictions=[str] and references=[[str]]
        bleu_score = bleu.compute(
            predictions=translations,
            references=[[ref] for ref in en_references]
        )["bleu"]

        #QA
        qa_outputs = []
        latencies_qa = []

        for s in it_sentences:
            prompt = (
                "Question: What is the main topic of the following text?\n"
                f"Text: {s}\nAnswer:"
            )
            out, latency = safe_generate(pipe, prompt, max_new_tokens=30)
            qa_outputs.append(out)
            if latency is not None:
                latencies_qa.append(latency)

        avg_latency_qa = sum(latencies_qa) / len(latencies_qa) if latencies_qa else None

        rouge_score = rouge.compute(
            predictions=qa_outputs,
            references=it_sentences
        )

        #SAVE 
        results.append({
            "Tokenizer": name,
            "BLEU": bleu_score,
            "ROUGE-1": rouge_score["rouge1"],
            "ROUGE-2": rouge_score["rouge2"],
            "ROUGE-L": rouge_score["rougeL"],
            "Avg_Latency_MT": avg_latency_mt,
            "Avg_Latency_QA": avg_latency_qa,
            "MT_Sample": " | ".join(translations[:NUM_SAMPLES_TO_SAVE]),
            "QA_Sample": " | ".join(qa_outputs[:NUM_SAMPLES_TO_SAVE]),
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Tokenizer", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
            "Avg_Latency_MT", "Avg_Latency_QA",
            "MT_Sample", "QA_Sample"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
