# /src/sava_adaptation.py

import torch
import numpy as np
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import lstsq
from tqdm import tqdm

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.1"
MINERVA_MODEL = "sapienzanlp/Minerva-3B-base-v1.0"
OUTPUT_DIR = "sava_output"

# Assicura che la cartella sia creata se questo file viene eseguito da solo
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# ----------------------------------------------------
# 1. FUNZIONI DI UTILITÀ (Caricamento Modelli)
# ----------------------------------------------------

def load_models_and_embeddings(model_s_id, model_h_id):
    """Carica i modelli Sorgente e Helper e le loro matrici di embedding."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on: {device}")

    # Caricamento Modello Sorgente (Mistral)
    try:
        tokenizer_s = AutoTokenizer.from_pretrained(model_s_id)
        model_s = AutoModelForCausalLM.from_pretrained(
            model_s_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" 
        )
        E_s = model_s.get_input_embeddings().weight.data.float().cpu() 
    except Exception as e:
        print(f"Errore nel caricare il modello sorgente {model_s_id}: {e}")
        return None, None, None, None, None, None

    # Caricamento Modello Helper (Minerva)
    try:
        tokenizer_h = AutoTokenizer.from_pretrained(model_h_id)
        model_h = AutoModelForCausalLM.from_pretrained(
            model_h_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        E_h = model_h.get_input_embeddings().weight.data.float().cpu()
    except Exception as e:
        print(f"Errore nel caricare il modello helper {model_h_id}: {e}")
        return None, None, None, None, None, None

    # Pulizia della VRAM per le matrici di embedding
    del model_s, model_h
    gc.collect()
    torch.cuda.empty_cache()

    return tokenizer_s, E_s, tokenizer_h, E_h, E_s.shape[1], E_h.shape[1]

# [Inserisci qui le funzioni compute_sava_mapping e initialize_new_embeddings]
# --- Le funzioni compute_sava_mapping e initialize_new_embeddings
# --- devono essere copiate qui dal tuo file sorgente completo.

def compute_sava_mapping(E_s, V_s, E_h, V_h, d_s, d_h):
    """Calcola la mappatura lineare W (Least Squares)."""
    print("\n--- Starting SAVA Mapping Computation ---")
    shared_tokens = set(V_s.keys()) & set(V_h.keys())
    print(f"Found {len(shared_tokens)} shared tokens for regression.")
    if len(shared_tokens) < max(d_s, d_h):
        print("ATTENZIONE: Pochi token condivisi per una regressione stabile.")
    E_s_shared = torch.stack([E_s[V_s[token]] for token in tqdm(shared_tokens, desc="Extracting E_s shared")])
    E_h_shared = torch.stack([E_h[V_h[token]] for token in tqdm(shared_tokens, desc="Extracting E_h shared")])
    Y_target = E_s_shared.numpy()
    X_input = E_h_shared.numpy()
    X_with_bias = np.concatenate([X_input, np.ones((X_input.shape[0], 1))], axis=1)
    M_solution, *_ = lstsq(X_with_bias, Y_target)
    W_matrix = torch.from_numpy(M_solution[:-1, :]).float()
    b_vector = torch.from_numpy(M_solution[-1, :]).float()
    print("--- SAVA Mapping Complete ---")
    torch.save(W_matrix, os.path.join(OUTPUT_DIR, "W_matrix.pt"))
    torch.save(b_vector, os.path.join(OUTPUT_DIR, "b_vector.pt"))
    return W_matrix, b_vector

def initialize_new_embeddings(E_s, E_h, V_h, V_s, W_matrix, b_vector, d_s):
    """Applica la mappatura per inizializzare i nuovi embedding (verifica metodologica)."""
    new_target_tokens = set(V_h.keys()) - set(V_s.keys())
    E_h_new = torch.stack([E_h[V_h[token]] for token in tqdm(new_target_tokens, desc="Extracting E_h new")])
    E_t_new = E_h_new @ W_matrix + b_vector
    mean_std_new = E_t_new.std(dim=0).mean()
    mean_std_original = E_s.std(dim=0).mean()
    print(f"Numero di nuovi embedding inizializzati: {E_t_new.shape[0]}")
    print(f"Deviazione standard media (Nuovi SAVA): {mean_std_new:.4f}")
    print(f"Deviazione standard media (Mistral originale): {mean_std_original:.4f}")
    torch.save(E_t_new, os.path.join(OUTPUT_DIR, "E_t_new_sava.pt"))
    
# ----------------------------------------------------
# MAIN EXECUTION (per testare solo il modulo SAVA)
# ----------------------------------------------------
if __name__ == "__main__":
    tokenizer_s, E_s, tokenizer_h, E_h, d_s, d_h = load_models_and_embeddings(MISTRAL_MODEL, MINERVA_MODEL)
    if tokenizer_s is not None:
        V_s = tokenizer_s.get_vocab()
        V_h = tokenizer_h.get_vocab()
        W_matrix, b_vector = compute_sava_mapping(E_s, V_s, E_h, V_h, d_s, d_h)
        if W_matrix is not None:
            initialize_new_embeddings(E_s, E_h, V_h, V_s, W_matrix, b_vector, d_s)
        print("\nSAVA Adaptation script finished.")