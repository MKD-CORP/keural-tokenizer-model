# Keural Tokenizer

A multilingual SentencePiece tokenizer trained for Korean + English + Code, designed for the Keural MoE foundation model.

---

## Tokenizer Specs

| Property | Value |
|---|---|
| Type | SentencePiece Unigram |
| Vocab size | 131,072 (128K) |
| Languages | Korean, English, Code |
| Normalization | NFKC |
| Byte fallback | ✅ Enabled |
| Digit splitting | ✅ Enabled |
| Training corpus | 26.74 GB |
| Training time | ~67 minutes |

### Special Tokens

| Token | ID |
|---|---|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

---

## Quick Start

### Install

```bash
pip install sentencepiece
```

### Use the tokenizer

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/keural_tokenizer.model")

# Encode
tokens = sp.Encode("안녕하세요, Hello world!", out_type=int)
print(tokens)  # [token ids]

# Decode
text = sp.Decode(tokens)
print(text)  # 안녕하세요, Hello world!

# Piece-level (human readable)
pieces = sp.Encode("The future of AI", out_type=str)
print(pieces)  # ['▁The', '▁future', '▁of', '▁AI']
```

### Verify integrity

```bash
sha256sum -c tokenizer/tokenizer.sha256
```

---

## Train from Scratch

Follow these 3 steps in order.

---

### Step 1 — Collect Corpus

```bash
python scripts/step1_collect_corpus.py
```

**What it does:**
- Downloads from 13 multilingual dataset sources via HuggingFace
- English: FineWeb, DOLMA, C4, Wikipedia, ArXiv, PubMed, OpenWebMath, Gutenberg
- Korean: FineWeb2, C4-ko, CC100-ko, Wikipedia-ko
- Code: The Stack, StarCoder
- Target: ~54GB, ~5.7M documents
- Saves to: `data/raw/tokenizer_corpus_parts/` then merges to `data/raw/tokenizer_corpus.txt`
- Supports resume (progress tracked in `data/logs/tokenizer_progress.json`)

**Requirements:**
```bash
pip install datasets huggingface_hub tqdm
```

**Minimum recommended corpus:**
- Size: 10GB+
- Documents: 500K+
- Languages: Must include both Korean and English for bilingual tokenizer

---

### Step 2 — Clean Corpus

```bash
python scripts/step2_clean_corpus.py \
  --input data/raw/tokenizer_corpus.txt \
  --output data/raw/tokenizer_corpus_clean.txt
```

**What it does:**
- Removes control characters and normalizes whitespace
- Deduplication using Blake2b hashing (handles up to 50M unique documents)
- Removes boilerplate (cookie notices, privacy policies, JS errors)
- Filters: min length 80 chars, max 5 URLs, alphanumeric ratio >25%
- Repetition filter: rejects texts with 10+ repeated characters
- Multi-threaded: uses all available CPU cores
- Reports: kept %, duplicates removed, throughput

**Expected output:**
- ~80-90% of input kept after cleaning
- Corpus ready for tokenizer training

---

### Step 3 — Train Tokenizer

```bash
python scripts/step3_train_tokenizer.py \
  --input data/raw/tokenizer_corpus_clean.txt \
  --output_dir my_tokenizer/ \
  --vocab_size 131072
```

**What it does:**
- Validates corpus before training (size, line count, language mix)
- Trains SentencePiece Unigram model
- Monitors CPU/RAM usage during training (logs every 10 seconds)
- Post-training validation on English, Korean, code, and mixed inputs
- Saves: `.model`, `.vocab`, `tokenizer_config.json`, `tokenizer_metadata.json`, `tokenizer.sha256`

**Key training parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `model_type` | `unigram` | Better than BPE for Korean morphology |
| `vocab_size` | `131072` | Large enough for Korean + English + Code |
| `character_coverage` | `0.9995` | 99.95% — covers rare Korean characters |
| `byte_fallback` | `true` | Never produces unknown tokens |
| `split_digits` | `true` | Better number handling |
| `normalization_rule_name` | `nfkc` | Standard Unicode normalization |

**Hardware requirements:**
- RAM: 64GB+ recommended (SentencePiece loads full corpus into memory)
- CPU: 8+ cores (training is CPU-only, no GPU needed)
- Disk: 3x corpus size free space
- Time: ~1-2 hours for 26GB corpus on 32 cores

---

## Why Unigram over BPE?

| | Unigram | BPE |
|---|---|---|
| Korean morphology | ✅ Better | ❌ Over-segments |
| Reversibility | ✅ Always | ✅ Always |
| Vocab optimization | ✅ Probabilistic | ❌ Greedy |
| Training speed | ❌ Slower | ✅ Faster |
| Used by | Llama, Mistral, T5 | GPT-2, GPT-4 |

Korean is an agglutinative language — words are formed by combining morphemes. Unigram handles this better than BPE by learning the optimal tokenization probabilistically.

---

## Why 128K vocab?

| Model | Vocab size |
|---|---|
| GPT-2 | 50,257 |
| LLaMA 2 | 32,000 |
| LLaMA 3 | 128,256 |
| Gemma | 256,000 |
| **Keural** | **131,072** |

128K is the sweet spot for Korean+English bilingual models:
- Enough tokens for Korean morphemes without fragmentation
- Matches LLaMA 3's vocab size range
- English tokenization efficiency: ~4.15 chars/token
- Korean tokenization: round-trip verified ✅

---

## File Structure

```
keural-tokenizer/
├── README.md
├── tokenizer/
│   ├── keural_tokenizer.model     # SentencePiece model binary (2.6MB)
│   ├── keural_tokenizer.vocab     # Human-readable vocabulary (2.5MB)
│   ├── tokenizer_config.json      # HuggingFace-compatible config
│   ├── tokenizer_metadata.json    # Full training metadata + validation results
│   └── tokenizer.sha256           # SHA256 hash for integrity verification
└── scripts/
    ├── step1_collect_corpus.py    # Download and collect training corpus
    ├── step2_clean_corpus.py      # Clean and deduplicate corpus
    └── step3_train_tokenizer.py   # Train SentencePiece tokenizer
```

---

## Integration with HuggingFace

```python
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm

# Load via SentencePiece directly
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/keural_tokenizer.model")

# Vocab size
print(sp.GetPieceSize())  # 131072

# BOS/EOS handling
bos_id = sp.PieceToId("<bos>")  # 1
eos_id = sp.PieceToId("<eos>")  # 2

# Encode with special tokens
def encode_for_training(text, max_len=4096):
    ids = [bos_id] + sp.Encode(text, out_type=int) + [eos_id]
    return ids[:max_len]
```

---

## License

This tokenizer was trained from scratch by **Md Najmul Hossain** as part of the Keural foundation model project. The tokenizer model, vocabulary, and training scripts are original work.

---

## Citation

```bibtex
@misc{keural-tokenizer-2026,
  author = {Md Najmul Hossain},
  title  = {Keural Tokenizer: A Multilingual SentencePiece Tokenizer for Korean and English},
  year   = {2026},
  url    = {https://github.com/mkd-hossain/keural-tokenizer}
}
```