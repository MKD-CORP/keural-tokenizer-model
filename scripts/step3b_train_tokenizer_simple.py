#!/usr/bin/env python3
"""
Keural-13B Foundation Tokenizer Trainer
Strict isolated project mode.
Aligned with architecture vocab_size=131072.
Production-safe for 50GB+ corpus.
"""

import os
import sys
import time
import hashlib
import multiprocessing
import sentencepiece as spm


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "tokenizer_corpus.txt")

TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
MODEL_PREFIX = os.path.join(TOKENIZER_DIR, "keural_tokenizer")

VOCAB_SIZE = 131072
SEED = 42

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

CHARACTER_COVERAGE = 0.9995
INPUT_SENTENCE_SIZE = 50_000_000   # safer for 50GB corpus
MODEL_TYPE = "unigram"

NUM_THREADS = min(32, multiprocessing.cpu_count())


# =====================================================
# PRE-CHECKS
# =====================================================

if not os.path.exists(RAW_FILE):
    print("ERROR: tokenizer_corpus.txt not found.")
    sys.exit(1)

os.makedirs(TOKENIZER_DIR, exist_ok=True)

model_path = MODEL_PREFIX + ".model"
vocab_path = MODEL_PREFIX + ".vocab"

if os.path.exists(model_path):
    print("ERROR: tokenizer already exists.")
    print("Delete tokenizer directory to retrain.")
    sys.exit(1)

file_size_gb = os.path.getsize(RAW_FILE) / (1024 ** 3)

print("=" * 60)
print("Keural-13B Tokenizer Training")
print("Corpus:", RAW_FILE)
print(f"Corpus size: {file_size_gb:.2f} GB")
print("Vocab size:", VOCAB_SIZE)
print("Threads:", NUM_THREADS)
print("Output:", TOKENIZER_DIR)
print("=" * 60)


# =====================================================
# TRAINING
# =====================================================

start_time = time.time()

spm.SentencePieceTrainer.train(
    input=RAW_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,

    character_coverage=CHARACTER_COVERAGE,
    byte_fallback=True,

    split_digits=False,
    add_dummy_prefix=False,
    remove_extra_whitespaces=True,
    normalization_rule_name="nfkc",

    input_sentence_size=INPUT_SENTENCE_SIZE,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,

    pad_id=PAD_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    unk_id=UNK_ID,

    pad_piece="<pad>",
    bos_piece="<bos>",
    eos_piece="<eos>",
    unk_piece="<unk>",

    seed=SEED,
    num_threads=NUM_THREADS
)

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed/60:.2f} minutes")


# =====================================================
# VALIDATION
# =====================================================

if not os.path.exists(model_path):
    print("ERROR: tokenizer.model not generated.")
    sys.exit(1)

sp = spm.SentencePieceProcessor()
sp.load(model_path)

assert sp.vocab_size() == VOCAB_SIZE, "Vocab size mismatch"
assert sp.pad_id() == PAD_ID, "PAD ID mismatch"
assert sp.bos_id() == BOS_ID, "BOS ID mismatch"
assert sp.eos_id() == EOS_ID, "EOS ID mismatch"

print("Tokenizer validation passed.")


# =====================================================
# HASH LOCK (IMMUTABILITY)
# =====================================================

def sha256sum(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

model_hash = sha256sum(model_path)
print("Tokenizer SHA256:", model_hash)


print("\nFinal files:")
print(" -", model_path)
print(" -", vocab_path)

print("\nIMPORTANT:")
print("Back up tokenizer directory now.")
print("Tokenizer must NEVER change after pretraining begins.")