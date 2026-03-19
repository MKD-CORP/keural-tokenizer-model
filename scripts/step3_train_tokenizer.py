#!/usr/bin/env python3
"""
Keural-13B Foundation Tokenizer Trainer
Production-grade tokenizer training script

Features
- Auto-detects the best corpus file
- Preflight validation and corpus stats
- Live CPU / RAM / thread monitoring during training
- SentencePiece Unigram tokenizer training
- Post-training validation
- SHA256 immutability lock
- HuggingFace-compatible tokenizer config export

Recommended corpus priority:
1) data/raw/tokenizer_corpus_final.txt
2) data/raw/tokenizer_corpus_shuffled.txt
3) data/raw/tokenizer_corpus_clean.txt
"""

import os
import sys
import json
import time
import math
import queue
import signal
import hashlib
import logging
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime

import psutil
import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
LOG_DIR = BASE_DIR / "data" / "logs"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_CANDIDATES = [
    DATA_DIR / "tokenizer_corpus_final.txt",
    DATA_DIR / "tokenizer_corpus_shuffled.txt",
    DATA_DIR / "tokenizer_corpus_clean.txt",
]

MODEL_PREFIX = TOKENIZER_DIR / "keural_tokenizer"

VOCAB_SIZE = 131072
MODEL_TYPE = "unigram"
CHARACTER_COVERAGE = 0.9995
NUM_THREADS = min(32, multiprocessing.cpu_count())

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD_PIECE = "<pad>"
BOS_PIECE = "<bos>"
EOS_PIECE = "<eos>"
UNK_PIECE = "<unk>"

MAX_SENTENCE_LENGTH = 10000
MONITOR_INTERVAL_SEC = 10
SAMPLE_LINES_FOR_ESTIMATE = 20000

TRAIN_EXTREMELY_LARGE_CORPUS = True
SHUFFLE_INPUT_SENTENCE = True
BYTE_FALLBACK = True
SPLIT_DIGITS = True
SPLIT_BY_UNICODE_SCRIPT = True
ADD_DUMMY_PREFIX = False
REMOVE_EXTRA_WHITESPACES = True
NORMALIZATION_RULE_NAME = "nfkc"

# use all lines if corpus is not too huge; otherwise cap at 10M
MAX_INPUT_SENTENCE_SIZE = 10_000_000

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_LOG_FILE = LOG_DIR / f"tokenizer_train_{RUN_TS}.log"
RESOURCE_LOG_FILE = LOG_DIR / f"tokenizer_resources_{RUN_TS}.jsonl"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(TRAIN_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("keural_tokenizer_trainer")


# ============================================================
# HELPERS
# ============================================================

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_corpus_file() -> Path:
    for path in CORPUS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No tokenizer corpus file found. Expected one of:\n"
        + "\n".join(str(p) for p in CORPUS_CANDIDATES)
    )


def count_lines_and_chars(path: Path):
    total_lines = 0
    total_chars = 0
    short_lines = 0
    long_lines = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            line_len = len(line.rstrip("\n"))
            total_chars += line_len

            if line_len < 80:
                short_lines += 1
            if line_len > MAX_SENTENCE_LENGTH:
                long_lines += 1

            if total_lines % 1_000_000 == 0:
                logger.info(
                    f"Corpus scan progress: {total_lines:,} lines | "
                    f"{total_chars / (1024**3):.2f} GB text observed"
                )

    return {
        "total_lines": total_lines,
        "total_chars": total_chars,
        "short_lines_lt_80": short_lines,
        "long_lines_gt_max_sentence_length": long_lines,
    }


def estimate_chars_per_token(path: Path) -> float:
    sample_chars = 0
    sample_lines = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            sample_chars += len(text)
            sample_lines += 1
            if sample_lines >= SAMPLE_LINES_FOR_ESTIMATE:
                break

    # conservative default for mixed KR/EN/code corpora
    if sample_lines == 0:
        return 4.0

    avg_doc_len = sample_chars / sample_lines

    if avg_doc_len < 300:
        return 3.5
    if avg_doc_len < 1000:
        return 3.8
    return 4.0


def pretty_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GB"


def choose_input_sentence_size(total_lines: int) -> int:
    return min(total_lines, MAX_INPUT_SENTENCE_SIZE)


# ============================================================
# RESOURCE MONITOR
# ============================================================

class ResourceMonitor(threading.Thread):
    def __init__(self, interval_sec: int = 10):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.process = psutil.Process(os.getpid())
        self.started_at = time.time()

    def stop(self):
        self.stop_event.set()

    def run(self):
        # warm-up cpu_percent
        self.process.cpu_percent(interval=None)

        with open(RESOURCE_LOG_FILE, "a", encoding="utf-8") as out:
            while not self.stop_event.is_set():
                try:
                    rss = self.process.memory_info().rss
                    vms = self.process.memory_info().vms
                    cpu = self.process.cpu_percent(interval=None)
                    threads = self.process.num_threads()
                    elapsed = time.time() - self.started_at
                    io = self.process.io_counters()

                    payload = {
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_sec": round(elapsed, 2),
                        "cpu_percent_process": round(cpu, 2),
                        "rss_bytes": rss,
                        "vms_bytes": vms,
                        "rss_gb": round(rss / (1024**3), 3),
                        "vms_gb": round(vms / (1024**3), 3),
                        "threads": threads,
                        "read_mb": round(io.read_bytes / (1024**2), 2),
                        "write_mb": round(io.write_bytes / (1024**2), 2),
                    }
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out.flush()

                    logger.info(
                        "[monitor] "
                        f"elapsed={elapsed/60:.1f} min | "
                        f"cpu={cpu:.1f}% | "
                        f"rss={rss/(1024**3):.2f} GB | "
                        f"threads={threads}"
                    )
                except Exception as e:
                    logger.warning(f"[monitor] failed: {e}")

                self.stop_event.wait(self.interval_sec)


# ============================================================
# PRECHECKS
# ============================================================

def precheck_output_files():
    model_path = Path(str(MODEL_PREFIX) + ".model")
    vocab_path = Path(str(MODEL_PREFIX) + ".vocab")

    if model_path.exists() or vocab_path.exists():
        logger.error("Tokenizer output already exists.")
        logger.error(f"Delete existing files under: {TOKENIZER_DIR}")
        sys.exit(1)


def run_preflight(corpus_path: Path):
    logger.info("=" * 70)
    logger.info("PRE-FLIGHT CHECK")
    logger.info("=" * 70)
    logger.info(f"Selected corpus: {corpus_path}")
    logger.info(f"Corpus size on disk: {pretty_gb(corpus_path.stat().st_size)}")

    stats = count_lines_and_chars(corpus_path)
    input_sentence_size = choose_input_sentence_size(stats["total_lines"])
    chars_per_token_est = estimate_chars_per_token(corpus_path)
    est_tokens = int(stats["total_chars"] / chars_per_token_est)

    logger.info(f"Total lines: {stats['total_lines']:,}")
    logger.info(f"Total chars: {stats['total_chars']:,}")
    logger.info(f"Short lines (<80): {stats['short_lines_lt_80']:,}")
    logger.info(
        f"Long lines (>{MAX_SENTENCE_LENGTH} chars): "
        f"{stats['long_lines_gt_max_sentence_length']:,}"
    )
    logger.info(f"Estimated chars/token: {chars_per_token_est:.2f}")
    logger.info(f"Estimated total tokens: {est_tokens:,}")
    logger.info(f"Input sentence size for trainer: {input_sentence_size:,}")
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info("=" * 70)

    if stats["total_lines"] < 100_000:
        logger.error("Corpus is too small for a 131072 tokenizer.")
        sys.exit(1)

    return stats, input_sentence_size, est_tokens


# ============================================================
# TRAINING
# ============================================================

def train_tokenizer(corpus_path: Path, input_sentence_size: int):
    logger.info("=" * 70)
    logger.info("TOKENIZER TRAINING START")
    logger.info("=" * 70)

    start_time = time.time()
    monitor = ResourceMonitor(interval_sec=MONITOR_INTERVAL_SEC)
    monitor.start()

    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(MODEL_PREFIX),
            vocab_size=VOCAB_SIZE,
            model_type=MODEL_TYPE,

            character_coverage=CHARACTER_COVERAGE,
            byte_fallback=BYTE_FALLBACK,
            split_digits=SPLIT_DIGITS,
            split_by_unicode_script=SPLIT_BY_UNICODE_SCRIPT,
            add_dummy_prefix=ADD_DUMMY_PREFIX,
            remove_extra_whitespaces=REMOVE_EXTRA_WHITESPACES,
            normalization_rule_name=NORMALIZATION_RULE_NAME,

            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
            train_extremely_large_corpus=TRAIN_EXTREMELY_LARGE_CORPUS,
            max_sentence_length=MAX_SENTENCE_LENGTH,
            num_threads=NUM_THREADS,

            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            unk_id=UNK_ID,

            pad_piece=PAD_PIECE,
            bos_piece=BOS_PIECE,
            eos_piece=EOS_PIECE,
            unk_piece=UNK_PIECE,

            # keep exact vocab target
            hard_vocab_limit=True,
        )
    finally:
        monitor.stop()
        monitor.join(timeout=5)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"TOKENIZER TRAINING COMPLETE | elapsed={elapsed/60:.2f} minutes")
    logger.info("=" * 70)
    return elapsed


# ============================================================
# VALIDATION
# ============================================================

def validate_tokenizer():
    model_path = Path(str(MODEL_PREFIX) + ".model")
    vocab_path = Path(str(MODEL_PREFIX) + ".vocab")

    if not model_path.exists():
        logger.error("Tokenizer model file not found after training.")
        sys.exit(1)

    if not vocab_path.exists():
        logger.error("Tokenizer vocab file not found after training.")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    assert sp.vocab_size() == VOCAB_SIZE, (
        f"Vocab size mismatch: expected {VOCAB_SIZE}, got {sp.vocab_size()}"
    )
    assert sp.pad_id() == PAD_ID
    assert sp.bos_id() == BOS_ID
    assert sp.eos_id() == EOS_ID
    assert sp.unk_id() == UNK_ID

    test_cases = {
        "korean": "안녕하세요 저는 큐럴 토크나이저를 테스트합니다.",
        "english": "Hello, this is a tokenizer validation test for Keural.",
        "code": "def hello_world():\n    return 42",
        "mixed": "안녕하세요 world 123 def test(): return True",
        "rare_char": "𠜎",
    }

    validation = {}

    for key, text in test_cases.items():
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)

        validation[key] = {
            "input": text,
            "num_tokens": len(ids),
            "pieces_preview": pieces[:20],
            "roundtrip_ok": decoded == text,
        }

    # English token efficiency
    english_text = test_cases["english"]
    english_ids = sp.encode(english_text, out_type=int)
    english_chars_per_token = len(english_text) / max(len(english_ids), 1)

    logger.info("=" * 70)
    logger.info("POST-TRAIN VALIDATION")
    logger.info("=" * 70)
    logger.info(f"✓ vocab size: {sp.vocab_size():,}")
    logger.info(
        f"✓ special tokens: PAD={sp.pad_id()}, BOS={sp.bos_id()}, "
        f"EOS={sp.eos_id()}, UNK={sp.unk_id()}"
    )
    logger.info(f"✓ Korean round-trip: {validation['korean']['roundtrip_ok']}")
    logger.info(f"✓ English round-trip: {validation['english']['roundtrip_ok']}")
    logger.info(f"✓ Code tokenization: {validation['code']['num_tokens']} tokens")
    logger.info(f"✓ Rare char tokenization: {validation['rare_char']['num_tokens']} tokens")
    logger.info(f"✓ English chars/token: {english_chars_per_token:.2f}")
    logger.info("=" * 70)

    if not validation["korean"]["roundtrip_ok"]:
        logger.error("Korean round-trip validation failed.")
        sys.exit(1)

    if not validation["english"]["roundtrip_ok"]:
        logger.error("English round-trip validation failed.")
        sys.exit(1)

    return {
        "english_chars_per_token": round(english_chars_per_token, 2),
        "validation_cases": validation,
    }


# ============================================================
# EXPORT METADATA
# ============================================================

def export_metadata(corpus_path: Path, corpus_stats: dict, est_tokens: int, elapsed_sec: float, validation_info: dict):
    model_path = Path(str(MODEL_PREFIX) + ".model")
    vocab_path = Path(str(MODEL_PREFIX) + ".vocab")
    model_hash = sha256sum(model_path)

    tokenizer_metadata = {
        "model_name": "Keural-13B Tokenizer",
        "status": "LOCKED - DO NOT MODIFY AFTER PRETRAINING",
        "created_at": datetime.now().isoformat(),

        "corpus": {
            "file": str(corpus_path),
            "size_gb": round(corpus_path.stat().st_size / (1024**3), 2),
            "total_lines": corpus_stats["total_lines"],
            "total_chars": corpus_stats["total_chars"],
            "estimated_tokens": est_tokens,
            "short_lines_lt_80": corpus_stats["short_lines_lt_80"],
            "long_lines_gt_max_sentence_length": corpus_stats["long_lines_gt_max_sentence_length"],
        },

        "tokenizer": {
            "model_type": MODEL_TYPE,
            "vocab_size": VOCAB_SIZE,
            "character_coverage": CHARACTER_COVERAGE,
            "byte_fallback": BYTE_FALLBACK,
            "split_digits": SPLIT_DIGITS,
            "split_by_unicode_script": SPLIT_BY_UNICODE_SCRIPT,
            "normalization_rule_name": NORMALIZATION_RULE_NAME,
            "max_sentence_length": MAX_SENTENCE_LENGTH,
            "num_threads": NUM_THREADS,
            "train_extremely_large_corpus": TRAIN_EXTREMELY_LARGE_CORPUS,
            "shuffle_input_sentence": SHUFFLE_INPUT_SENTENCE,
        },

        "special_tokens": {
            "pad": {"id": PAD_ID, "piece": PAD_PIECE},
            "bos": {"id": BOS_ID, "piece": BOS_PIECE},
            "eos": {"id": EOS_ID, "piece": EOS_PIECE},
            "unk": {"id": UNK_ID, "piece": UNK_PIECE},
        },

        "training": {
            "elapsed_minutes": round(elapsed_sec / 60, 2),
            "train_log_file": str(TRAIN_LOG_FILE),
            "resource_log_file": str(RESOURCE_LOG_FILE),
        },

        "validation": validation_info,

        "immutability": {
            "sha256_model": model_hash,
            "model_file": str(model_path),
            "vocab_file": str(vocab_path),
        },
    }

    metadata_file = TOKENIZER_DIR / "tokenizer_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(tokenizer_metadata, f, indent=2, ensure_ascii=False)

    hash_file = TOKENIZER_DIR / "tokenizer.sha256"
    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(f"{model_hash}  {model_path.name}\n")

    hf_config = {
        "architectures": ["KeuralMoEForCausalLM"],
        "model_type": "keural_moe",
        "vocab_size": VOCAB_SIZE,
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID,
        "unk_token_id": UNK_ID,
        "tokenizer_class": "SentencePieceTokenizer",
        "sentencepiece_model_file": model_path.name,
        "special_tokens_map": {
            "pad_token": PAD_PIECE,
            "bos_token": BOS_PIECE,
            "eos_token": EOS_PIECE,
            "unk_token": UNK_PIECE,
        },
        "normalization": NORMALIZATION_RULE_NAME,
        "split_digits": SPLIT_DIGITS,
        "byte_fallback": BYTE_FALLBACK,
        "max_context_target": 1_048_576,
        "context_stages": [
            4096,
            8192,
            32768,
            131072,
            262144,
            524288,
            1048576,
        ],
    }

    hf_config_file = TOKENIZER_DIR / "tokenizer_config.json"
    with open(hf_config_file, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("IMMUTABILITY LOCK")
    logger.info("=" * 70)
    logger.info(f"SHA256: {model_hash}")
    logger.info("Files created:")
    logger.info(f"  - {model_path}")
    logger.info(f"  - {vocab_path}")
    logger.info(f"  - {metadata_file}")
    logger.info(f"  - {hash_file}")
    logger.info(f"  - {hf_config_file}")
    logger.info("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    def handle_sigint(signum, frame):
        logger.warning("Interrupted. Exiting safely.")
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    corpus_path = select_corpus_file()
    precheck_output_files()
    corpus_stats, input_sentence_size, est_tokens = run_preflight(corpus_path)
    elapsed = train_tokenizer(corpus_path, input_sentence_size)
    validation_info = validate_tokenizer()
    export_metadata(corpus_path, corpus_stats, est_tokens, elapsed, validation_info)

    logger.info("")
    logger.info("Tokenizer training is complete.")
    logger.info("Back up the entire tokenizer directory now.")
    logger.info("Do not change this tokenizer after pretraining starts.")
    logger.info("")


if __name__ == "__main__":
    main()


# tokenizer model training script 