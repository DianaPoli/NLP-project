
# sava_train.py

import torch
import numpy as np
import os
import gc
import random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets, DatasetDict
from trl import SFTTrainer
from tqdm import tqdm
import bitsandbytes
from typing import Dict, List, Tuple
from latentis.transform.translate import Translator, Procrustes
from latentis.transform.translate.aligner import SGDAffineAligner, MatrixAligner, ZeroPadding
from latentis.transform.translate.functional import lstsq_align_state
from latentis.transform import TransformSequence
from latentis.transform.base import StandardScaling, MeanLPNorm

# config
SEED = 42
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.1"
MINERVA_MODEL = "sapienzanlp/Minerva-3B-base-v1.0"
DATA_PATH_IT = "data/italian_corpus.txt"
DATA_PATH_EN = "data/english_corpus.txt"
OUTPUT_DIR_TRAIN = "./results/mct_sava_output"
TOTAL_TRAINING_SAMPLES = 500

# reproducibility
def reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

reproducibility(SEED)
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)

# aligners for SAVA initialization
ALIGNERS_embedding = {
    "matrix": lambda _: MatrixAligner(name="affine", align_fn_state=lstsq_align_state),
    "sgd": lambda seed: SGDAffineAligner(num_steps=1000, lr=1e-3, random_seed=seed),
    "ortho": lambda _: Procrustes()
}

# SAVA class from github repository

class SemanticAlignmentEmbeddingInitializer:
    def __init__(self, source_model, source_tokenizer, helper_model, target_tokenizer, seed=42, tie_weights=True, aligner="sgd", num_anchors=5000, anchor_selection="full"):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.helper_model = helper_model
        self.target_tokenizer = target_tokenizer
        self.seed = seed
        self.tie_weights = tie_weights
        self.aligner = aligner
        self.num_anchors = num_anchors
        self.anchor_selection = anchor_selection
        self.vocabulary_intersection = {}
        self.target_vocabulary_out = {}

        self.translator_embedding = Translator(
            aligner=ALIGNERS_embedding[self.aligner](seed),
            x_transform=TransformSequence([StandardScaling(), MeanLPNorm(p=2)]),
            y_transform=TransformSequence([StandardScaling(), MeanLPNorm(p=2)]),
            dim_matcher=ZeroPadding()
        )
        reproducibility(self.seed)

    def _map_llama_to_minerva(self, token_str: str):
        if token_str == "<|begin_of_text|>": return "<s>"
        if token_str == "<|end_of_text|>": return "</s>"
        return token_str.replace("Ġ", " ")

    def _full_intersection_anchors(self) -> list[tuple]:
        anchor_tokens = list(self.vocabulary_intersection.items())
        self.num_anchors = len(anchor_tokens)
        return anchor_tokens

    def get_spaces_intersection(self, helper_embeddings, source_embeddings):
        source_embeddings_intersection = torch.zeros((len(self.vocabulary_intersection), source_embeddings.size(1)))
        helper_embeddings_intersection = torch.zeros((len(self.vocabulary_intersection), helper_embeddings.size(1)))
        for i, (_, (source_idx, target_idx)) in enumerate(self.vocabulary_intersection.items()):
            source_embeddings_intersection[i] = source_embeddings[source_idx]
            helper_embeddings_intersection[i] = helper_embeddings[target_idx]
        return helper_embeddings_intersection, source_embeddings_intersection

    def extract_anchor_embeddings(self, anchor_tokens, helper_embeddings, source_embeddings):
        source_embeddings_anchor = torch.zeros((self.num_anchors, source_embeddings.size(1)))
        helper_embeddings_anchor = torch.zeros((self.num_anchors, helper_embeddings.size(1)))
        for i, (token, (source_idx, target_idx)) in enumerate(anchor_tokens):
            source_embeddings_anchor[i] = source_embeddings[source_idx]
            helper_embeddings_anchor[i] = helper_embeddings[target_idx]
        return helper_embeddings_anchor, source_embeddings_anchor

    def compute_transformation_embeddings(self, helper_embeddings, helper_embeddings_anchor,
                                          helper_embeddings_intersection, source_embeddings, source_embeddings_anchor,
                                          source_embeddings_intersection, source_token_to_idx, target_embeddings,
                                          target_token_to_idx, translator):
        translator.fit(x=helper_embeddings_anchor, y=source_embeddings_anchor)

        for token, target_idx in self.target_vocabulary_out.items():
            if token in source_token_to_idx:
                target_embeddings.weight.data[target_idx] = source_embeddings[source_token_to_idx[token]]
            else:
                translated = translator(helper_embeddings[target_token_to_idx[token]].unsqueeze(0))["x"].flatten()
                target_embeddings.weight.data[target_idx] = translated

        try:
            pred = translator(helper_embeddings_intersection)["x"]
            mse = (source_embeddings_intersection - pred).abs().mean()
            cos = F.cosine_similarity(source_embeddings_intersection, pred).abs().mean()
            print(f"MEAN SQUARED ERROR - EMBEDDING : {mse}")
            print(f"COSINE - EMBEDDING : {cos}")
        except Exception as e:
            print("Warning: diagnostic embedding failed:", e)


    def __call__(self) -> AutoModelForCausalLM:
        target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}
        source_token_to_idx = {self._map_llama_to_minerva(t): i for t, i in self.source_tokenizer.get_vocab().items()}

        self.target_vocabulary_out = target_token_to_idx.copy()
        for token, target_idx in target_token_to_idx.items():
            if token in source_token_to_idx:
                source_idx = source_token_to_idx[token]
                self.vocabulary_intersection[token] = (source_idx, target_idx)
        print(f"Intersection length : ({len(self.vocabulary_intersection)})")
        anchor_tokens = self._full_intersection_anchors()

        source_embeddings = self.source_model.get_input_embeddings().weight.detach().cpu()
        helper_embeddings = self.helper_model.get_input_embeddings().weight.detach().cpu()

        helper_embeddings_intersection, source_embeddings_intersection = self.get_spaces_intersection(
            helper_embeddings, source_embeddings)

        embedding_dim = source_embeddings.size(1)
        target_vocab_size = len(self.target_vocabulary_out)
        target_embeddings = torch.nn.Embedding(target_vocab_size, embedding_dim)

        helper_embeddings_anchor, source_embeddings_anchor = self.extract_anchor_embeddings(anchor_tokens, helper_embeddings, source_embeddings)

        self.compute_transformation_embeddings(helper_embeddings, helper_embeddings_anchor,
                                              helper_embeddings_intersection, source_embeddings,
                                              source_embeddings_anchor, source_embeddings_intersection,
                                              source_token_to_idx, target_embeddings, target_token_to_idx,
                                              self.translator_embedding)

        try:
            orig_embedding = self.source_model.get_input_embeddings()
            ref_device = getattr(orig_embedding.weight, "device", torch.device("cpu"))
            ref_dtype = getattr(orig_embedding.weight, "dtype", torch.float32)
        except Exception:
            ref_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ref_dtype = torch.float32

        try:
            self.source_model.resize_token_embeddings(target_vocab_size)
        except Exception as e:
            print("Warning: resize_token_embeddings failed, attempting fallback:", e)
            self.source_model.config.vocab_size = target_vocab_size

        model_emb = self.source_model.get_input_embeddings()
        model_emb_device = getattr(model_emb.weight, "device", ref_device)
        model_emb_dtype = getattr(model_emb.weight, "dtype", ref_dtype)

        cpu_weights = target_embeddings.weight.data  

        with torch.no_grad():
            try:
                if cpu_weights.shape == model_emb.weight.data.shape:
                    model_emb.weight.data.copy_(cpu_weights.to(device=model_emb_device, dtype=model_emb_dtype))
                else:
                    nrows = min(cpu_weights.shape[0], model_emb.weight.data.shape[0])
                    model_emb.weight.data[:nrows].copy_(cpu_weights[:nrows].to(device=model_emb_device, dtype=model_emb_dtype))
            except Exception:
                for i in range(min(cpu_weights.shape[0], model_emb.weight.data.shape[0])):
                    model_emb.weight.data[i] = cpu_weights[i].to(device=model_emb_device, dtype=model_emb_dtype)

        if hasattr(self.source_model, "lm_head"):
            try:
                if self.source_model.lm_head.weight.shape == model_emb.weight.shape:
                    self.source_model.lm_head.weight.data.copy_(model_emb.weight.data)
                else:
                    try:
                        self.source_model.lm_head.weight.data[:model_emb.weight.shape[0], :].copy_(model_emb.weight.data)
                    except Exception:
                        pass
            except Exception as e:
                print("Warning: could not sync lm_head weight (non-fatal):", e)

        if self.tie_weights:
            try:
                self.source_model.tie_weights()
            except Exception as e:
                print("Warning: tie_weights failed:", e)

        self.source_model.config.pad_token_id = self.target_tokenizer.pad_token_id
        if self.target_tokenizer.eos_token_id is not None:
            self.source_model.config.eos_token_id = self.target_tokenizer.eos_token_id

        try:
            emb_dev = self.source_model.get_input_embeddings().weight.device
            emb_dt = self.source_model.get_input_embeddings().weight.dtype
            print(f"SAVA: new input_embeddings on device={emb_dev}, dtype={emb_dt}")
        except Exception:
            pass

        return self.source_model


# mini - continual training

def load_models_and_init(model_s_id, model_h_id):
    """Load models, SAVA initialization execution and LoRA application"""

    # config bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # loading base models
    print(">> Loading base models (may download large files)...")
    model_s = AutoModelForCausalLM.from_pretrained(model_s_id, quantization_config=bnb_config, device_map="auto")
    model_h = AutoModelForCausalLM.from_pretrained(model_h_id, quantization_config=bnb_config, device_map="auto")
    tokenizer_s = AutoTokenizer.from_pretrained(model_s_id)
    tokenizer_h = AutoTokenizer.from_pretrained(model_h_id)

    # SAVA initialization
    print("\n--- SAVA initialization execution ---")

    sava_initializer = SemanticAlignmentEmbeddingInitializer(
        source_model=model_s, source_tokenizer=tokenizer_s, helper_model=model_h,
        target_tokenizer=tokenizer_h, aligner="sgd"
    )
    model_sava_initialized = sava_initializer()

    # forward-test verifying the absence of mismatch device/dtype
    try:
        model_device = next(model_sava_initialized.parameters()).device
        test_token_id = model_sava_initialized.config.eos_token_id if getattr(model_sava_initialized.config, "eos_token_id", None) is not None else 0
        dummy_input = torch.tensor([[test_token_id]], dtype=torch.long, device=model_device)
        with torch.no_grad():
            _ = model_sava_initialized(input_ids=dummy_input)
        print("Forward test after SAVA: OK")
    except Exception as e:
        print("Warning: forward test after SAVA failed:", e)

    # free helper model cleaning VRAM
    try:
        del model_h
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PEFT preparation
    print("\n--- Setup PEFT/LoRA ---")
    model_sava_initialized = prepare_model_for_kbit_training(model_sava_initialized)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model_sava_peft = get_peft_model(model_sava_initialized, lora_config)
    model_sava_peft.print_trainable_parameters()

    return model_sava_peft, tokenizer_h


def load_training_dataset(tokenizer):
    """Load and Preprocessing Dataset 75% IT / 25% EN"""

    # load dataset (IT, EN)
    if not os.path.exists(DATA_PATH_IT) or not os.path.exists(DATA_PATH_EN):
        print(f"ERROR: dataset files not found in {DATA_PATH_IT} or {DATA_PATH_EN}.")
        return DatasetDict({'train': load_dataset("text", data_files={"train": DATA_PATH_IT}, split="train").select(range(0))})

    italian_dataset = load_dataset("text", data_files={"train": DATA_PATH_IT}, split="train")
    english_dataset = load_dataset("text", data_files={"train": DATA_PATH_EN}, split="train")

    # dims (75% IT, 25% EN, budget max)
    MAX_IT_SAMPLES = len(italian_dataset)
    MAX_EN_SAMPLES = len(english_dataset)

    EN_SAMPLES = int(TOTAL_TRAINING_SAMPLES * 0.25)
    IT_SAMPLES = TOTAL_TRAINING_SAMPLES - EN_SAMPLES

    IT_SAMPLES = min(IT_SAMPLES, MAX_IT_SAMPLES)
    EN_SAMPLES = min(EN_SAMPLES, MAX_EN_SAMPLES)

    italian_subset = italian_dataset.select(range(IT_SAMPLES))
    english_subset = english_dataset.select(range(EN_SAMPLES))

    mixed_dataset = concatenate_datasets([italian_subset, english_subset])
    mixed_dataset = mixed_dataset.shuffle(seed=SEED)

    # tokenization and preprocessing
    tokenized_dataset = mixed_dataset.map(
        lambda samples: tokenizer(samples["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"]
    )

    dataset = tokenized_dataset.map(
        lambda samples: {"labels": samples["input_ids"].copy()},
        batched=True
    )

    print(f"M-CT Training Dataset ready with {len(dataset)} exampled mixed ({IT_SAMPLES} IT / {EN_SAMPLES} EN)")
    return DatasetDict({'train': dataset})


def main_mini_train():
    # 1. load and init
    model_sava, tokenizer_sava = load_models_and_init(MISTRAL_MODEL, MINERVA_MODEL)

    # 2. load training dataset 75/25
    dataset = load_training_dataset(tokenizer_sava)

    # 3. setup training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_TRAIN,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    # 4. trainer
    trainer = SFTTrainer(
        model=model_sava,
        args=training_args,
        train_dataset=dataset['train'],
    )

    print("\n--- starting mini-continual training (LoRA) ---")
    trainer.train()

    print("\n Mini-Continual Training SAVA completed. Results in:", OUTPUT_DIR_TRAIN)


if __name__ == "__main__":
    main_mini_train()