# Keural Tokenizer

> **Custom multilingual SentencePiece tokenizer for Korean + English + Code**
> Designed for the Keural MoE foundation model. Vocab size: 131,072.
> Built and maintained by **Md Najmul Hossain** / **MKD CO., LTD.**

---

## Table of Contents

1. [What is This?](#what-is-this)
2. [Tokenizer Specs](#tokenizer-specs)
3. [Quick Start](#quick-start)
4. [Compatibility](#compatibility)
5. [Train from Scratch](#train-from-scratch)
6. [Design Decisions](#design-decisions)
7. [Future Roadmap](#future-roadmap)
8. [Author & License](#author--license)

---

## What is This?

This is a **custom SentencePiece Unigram tokenizer** trained from scratch on a 26.74GB multilingual corpus. It is the official tokenizer for the Keural foundation model.

**It is NOT based on any existing tokenizer.** It was trained independently on Korean + English + Code text, optimized specifically for the Korean language morphology.

> ⚠️ **IMPORTANT:** This tokenizer is **locked after pretraining begins**. Never retrain or replace it — doing so will break compatibility with all existing checkpoints.

---

## Tokenizer Specs

### Core Properties

| Property | Value | Notes |
|---|---|---|
| Type | SentencePiece Unigram | Better than BPE for Korean |
| Vocab size | **131,072** | Must never change |
| Languages | Korean, English, Code | Primary: Korean |
| Normalization | NFKC | Unicode compatibility |
| Byte fallback | ✅ Enabled | Never produces `<unk>` |
| Digit splitting | ✅ Enabled | Better number handling |
| Training corpus | 26.74 GB | 4.3M documents |
| Training time | ~67 minutes | 32 CPU cores |

### Special Token IDs (LOCKED — NEVER CHANGE)

| Token | ID | Usage |
|---|---|---|
| `<pad>` | **0** | Padding token |
| `<bos>` | **1** | Beginning of sequence |
| `<eos>` | **2** | End of sequence |
| `<unk>` | **3** | Unknown (fallback, rarely used) |

> These IDs are hardcoded into the model architecture and training pipeline. Any mismatch will produce wrong results silently.

### Integrity Verification

```bash
sha256sum -c tokenizer/tokenizer.sha256
# Expected: keural_tokenizer.model: OK
```

SHA256: `b982818ea2f2057ba791e2006d17683799f1d8ceb9c91322018a638c4ec4b170`

### Training Corpus Breakdown

| Source | Language | Size |
|---|---|---|
| FineWeb | English | ~8GB |
| C4 | English/Korean | ~6GB |
| CC100 Korean | Korean | ~3GB |
| Korean WebText | Korean | ~3GB |
| Wikipedia (KO+EN) | Korean + English | ~2GB |
| ArXiv | English Science | ~2GB |
| PubMed | English Medical | ~1GB |
| The Stack | Code | ~1GB |
| Other | Mixed | ~0.74GB |
| **Total** | | **26.74GB** |

### Tokenization Efficiency

| Language | Chars/token | Example |
|---|---|---|
| English | ~4.15 | `"Hello world"` → 2 tokens |
| Korean | ~2.8 | `"안녕하세요"` → 3 tokens |
| Code | ~3.5 | `"def foo():"` → 4 tokens |

---

## Quick Start

### Install

```bash
pip install sentencepiece
```

### Load tokenizer

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/keural_tokenizer.model")

print(f"Vocab size: {sp.GetPieceSize()}")  # 131072
```

### Encode text

```python
# Encode to token IDs
text = "안녕하세요, Hello world!"
ids = sp.Encode(text, out_type=int)
print(ids)  # [12345, 678, 90, ...]

# Encode to piece strings (human readable)
pieces = sp.Encode(text, out_type=str)
print(pieces)  # ['▁안녕하세요', ',', '▁Hello', '▁world', '!']
```

### Decode tokens

```python
ids = [12345, 678, 90]
text = sp.Decode(ids)
print(text)  # reconstructed text
```

### Encode for training (with BOS/EOS)

```python
BOS_ID = 1  # <bos>
EOS_ID = 2  # <eos>
PAD_ID = 0  # <pad>

def encode_for_training(text: str, max_len: int = 4096) -> list:
    """Encode text with BOS/EOS for model training."""
    ids = [BOS_ID] + sp.Encode(text, out_type=int) + [EOS_ID]
    return ids[:max_len]

def encode_for_inference(text: str) -> list:
    """Encode prompt for inference (BOS only, no EOS)."""
    return [BOS_ID] + sp.Encode(text, out_type=int)
```

### Verify integrity

```bash
sha256sum -c tokenizer/tokenizer.sha256
```

---

## Compatibility

### Python & Library Versions

| Library | Tested version | Install |
|---|---|---|
| Python | 3.12.x | — |
| sentencepiece | 0.2.0+ | `pip install sentencepiece` |
| PyTorch | 2.6.0+ | (for model training only) |

### Model Compatibility

This tokenizer is the **only** tokenizer compatible with Keural model checkpoints. Using any other tokenizer will produce incorrect results.

| Repository | Compatible |
|---|---|
| [Keural-Model-Training](https://github.com/mkd-hossain/Keural-Model-Training) | ✅ Required |
| Any other model | ❌ Not compatible |

### Integration with HuggingFace Transformers

```python
import sentencepiece as spm

class KeuralTokenizer:
    """Minimal wrapper for use in training pipelines."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()  # 131072
        self.bos_id = self.sp.PieceToId("<bos>")  # 1
        self.eos_id = self.sp.PieceToId("<eos>")  # 2
        self.pad_id = self.sp.PieceToId("<pad>")  # 0
        self.unk_id = self.sp.PieceToId("<unk>")  # 3

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        ids = self.sp.Encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.Decode(ids)

    def __len__(self) -> int:
        return self.vocab_size

# Usage
tokenizer = KeuralTokenizer("tokenizer/keural_tokenizer.model")
ids = tokenizer.encode("안녕하세요!")
text = tokenizer.decode(ids)
```

---

## Train from Scratch

If you want to train your own version of this tokenizer, follow these steps exactly.

### Requirements

```bash
pip install sentencepiece datasets huggingface_hub tqdm
```

**Hardware:**
- CPU: 8+ cores (tokenizer training is CPU-only, no GPU needed)
- RAM: 64GB+ (SentencePiece loads full corpus into memory)
- Disk: 3× corpus size free space
- Time: ~1-2 hours for 26GB corpus on 32 cores

---

### Step 1 — Collect Corpus

```bash
# Production grade (recommended)
python scripts/step1_collect_corpus.py

# Simple version (fewer sources)
python scripts/step1b_collect_corpus_simple.py

# Streaming version (low memory)
python scripts/step1c_collect_corpus_streaming.py
```

**Recommended script:** `step1_collect_corpus.py` (production grade)

**What it collects:**

| Source | Language | Target docs |
|---|---|---|
| FineWeb | English | 1,000,000 |
| FineWeb 2 | Korean | 500,000 |
| DOLMA | English | 600,000 |
| C4 (English) | English | 600,000 |
| C4 (Korean) | Korean | 600,000 |
| CC100 Korean | Korean | 300,000 |
| The Stack | Code | 400,000 |
| StarCoder | Code | 200,000 |
| Wikipedia EN | English | 300,000 |
| Wikipedia KO | Korean | 150,000 |
| ArXiv | Science | 300,000 |
| PubMed | Medical | 200,000 |
| OpenWebMath | Math | 200,000 |
| Gutenberg | Books | 250,000 |
| **Total** | | **~5.6M docs** |

**Output:** `data/raw/tokenizer_corpus.txt` (~54GB target)

> ✅ Script supports resume — safe to interrupt and restart

---

### Step 2 — Clean Corpus

```bash
python scripts/step2_clean_corpus.py \
  --input data/raw/tokenizer_corpus.txt \
  --output data/raw/tokenizer_corpus_clean.txt
```

**Cleaning rules applied:**

| Rule | Detail |
|---|---|
| Remove control characters | ASCII 0–31, 127–159 |
| Normalize whitespace | collapse spaces, remove tabs |
| Min length | 80 characters |
| Max URLs | 5 per document |
| Alphanumeric ratio | >25% |
| Repetition filter | No 10+ repeated chars |
| Boilerplate removal | Cookie notices, JS errors, privacy policies |
| Deduplication | Blake2b hashing, up to 50M unique docs |

**Performance:** Multi-threaded, uses all CPU cores. ~80-90% of input retained.

---

### Step 3 — Train Tokenizer

```bash
# Production grade (recommended — includes monitoring and validation)
python scripts/step3_train_tokenizer.py \
  --input data/raw/tokenizer_corpus_clean.txt \
  --output_dir my_tokenizer/ \
  --vocab_size 131072

# Simple version
python scripts/step3b_train_tokenizer_simple.py
```

**Key SentencePiece parameters:**

| Parameter | Value | Why |
|---|---|---|
| `model_type` | `unigram` | Better Korean morphology than BPE |
| `vocab_size` | `131072` | Large enough for bilingual + code |
| `character_coverage` | `0.9995` | Covers 99.95% of chars |
| `byte_fallback` | `true` | Never produces `<unk>` |
| `split_digits` | `true` | Better number handling |
| `normalization_rule_name` | `nfkc` | Unicode normalization |
| `shuffle_input_sentence` | `true` | Better training distribution |
| `train_extremely_large_corpus` | `true` | Required for >1GB corpus |
| `num_threads` | `32` | Use all CPU cores |

**Outputs:**
- `my_tokenizer/keural_tokenizer.model` — binary model (2.6MB)
- `my_tokenizer/keural_tokenizer.vocab` — human-readable vocab (2.5MB)
- `my_tokenizer/tokenizer_config.json` — HuggingFace-compatible config
- `my_tokenizer/tokenizer_metadata.json` — training metadata + validation
- `my_tokenizer/tokenizer.sha256` — integrity hash

---

### Step 4 — Build Binary Dataset (Optional)

After training the tokenizer, use it to build the binary training dataset:

```bash
python scripts/step4_build_binary_dataset.py \
  --input_dir data/raw_stage1 \
  --output_dir data/binary \
  --tokenizer my_tokenizer/keural_tokenizer.model \
  --seq_len 4096
```

See [Keural-Model-Training](https://github.com/mkd-hossain/Keural-Model-Training) for the full training pipeline.

---

## Design Decisions

### Why SentencePiece Unigram (not BPE)?

| Feature | Unigram | BPE |
|---|---|---|
| Korean morphology | ✅ Optimal segmentation | ❌ Over-segments agglutinative morphemes |
| Vocab optimization | ✅ Probabilistic EM algorithm | ❌ Greedy merging |
| Reversibility | ✅ Always lossless | ✅ Always lossless |
| Training speed | ❌ Slower | ✅ Faster |
| Used by | LLaMA 1/2/3, Mistral, T5, Gemma | GPT-2/3/4, Claude |

Korean is an **agglutinative language** — one word can carry the meaning of an entire English sentence through morpheme combinations. Unigram tokenization learns the probabilistically optimal segmentation, which handles Korean morphology significantly better than BPE's greedy merge approach.

### Why 131,072 vocab?

| Model | Vocab | Performance |
|---|---|---|
| LLaMA 2 | 32,000 | Poor Korean (over-segments) |
| Mistral 7B | 32,000 | Poor Korean |
| GPT-4 (estimate) | ~100,000 | Good multilingual |
| LLaMA 3 | 128,256 | Good multilingual |
| **Keural** | **131,072** | Optimized for Korean+English |
| Gemma | 256,000 | Over-parameterized for Korean |

At 32K vocab, Korean text uses 3-4× more tokens than English (inefficient). At 131K, Korean efficiency is comparable to English — both around 2.5-4.5 chars/token.

### Why NFKC Normalization?

NFKC (Unicode Normalization Form KC) decomposes characters into canonical equivalents and then recomposes. For Korean, this ensures that:
- Different encodings of the same Hangul syllable are treated identically
- Full-width/half-width character variants are unified
- Compatibility characters are normalized

---

## Future Roadmap

### Tokenizer Improvements (future versions)

| Version | Change | Reason |
|---|---|---|
| v1 (current) | 131,072 vocab, 4K context | Stage 1 pretraining |
| v2 (planned) | Add domain-specific tokens | Better science/medical/legal |
| v3 (planned) | Extended for 1M context | Ultra-long document support |

> ⚠️ Any new tokenizer version requires training a new model from scratch. The current tokenizer (v1) is permanent for all existing Keural checkpoints.

### Integration Roadmap

- [ ] HuggingFace `transformers` compatible wrapper
- [ ] `tokenizers` (Rust) fast tokenizer implementation
- [ ] Public release on HuggingFace Hub
- [ ] JavaScript/TypeScript tokenizer port (for web inference)

---

## File Structure

```
keural-tokenizer/
├── README.md
├── tokenizer/
│   ├── keural_tokenizer.model     # SentencePiece binary (2.6MB) ← USE THIS
│   ├── keural_tokenizer.vocab     # Human-readable vocabulary list (2.5MB)
│   ├── tokenizer_config.json      # HuggingFace-compatible config
│   ├── tokenizer_metadata.json    # Full training metadata + test results
│   └── tokenizer.sha256           # SHA256 integrity hash
└── scripts/
    ├── step1_collect_corpus.py         # Production corpus collector (14 sources)
    ├── step1b_collect_corpus_simple.py # Simple corpus collector
    ├── step1c_collect_corpus_streaming.py # Low-memory streaming collector
    ├── step2_clean_corpus.py           # Corpus cleaner + deduplicator
    ├── step3_train_tokenizer.py        # Production tokenizer trainer
    ├── step3b_train_tokenizer_simple.py # Simple tokenizer trainer
    └── step4_build_binary_dataset.py   # Binary dataset builder for model training
```

---

## Author & License

**Md Najmul Hossain**
CEO / Founder — MKD CO., LTD.
Email: hossain.najmul@mkd.kr

This tokenizer was built entirely from scratch as part of the Keural foundation model project. The training corpus, cleaning pipeline, and tokenizer configuration are original work.

Related repositories:
- Model Training: [github.com/mkd-hossain/Keural-Model-Training](https://github.com/mkd-hossain/Keural-Model-Training)

```bibtex
@misc{keural-tokenizer-2026,
  author = {Md Najmul Hossain},
  title  = {Keural Tokenizer: A Multilingual SentencePiece Tokenizer for Korean and English},
  year   = {2026},
  url    = {https://github.com/mkd-hossain/keural-tokenizer}
}
```