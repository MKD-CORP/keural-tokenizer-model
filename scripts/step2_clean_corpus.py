#!/usr/bin/env python3
"""
Keural Tokenizer Corpus Cleaner - Production Ready (High Performance)
Optimized for 480GB RAM, 32 cores, NVMe disk
"""

import re
import sys
import hashlib
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from functools import partial
from tqdm import tqdm

# Hardware optimization
NUM_WORKERS = min(mp.cpu_count(), 32)  # Cap at 32
CHUNK_SIZE = 1000  # Lines per worker chunk
WRITE_BUFFER = 16 * 1024 * 1024  # 16MB
MAX_HASHES = 50_000_000  # Lower for memory safety (~400MB)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"corpus_clean_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Filtering rules
BAD_PATTERNS = [
    "cookie policy", "privacy policy", "all rights reserved",
    "enable javascript", "sign up for", "newsletter",
    "terms of service", "terms of use", "subscribe",
    "unsubscribe", "advertisement", "sponsored",
]

CODE_PATTERNS = [
    "def ", "class ", "import ", "from ", "return ",
    "function", "const ", "let ", "var ", "public ",
]

# Compiled regex (faster)
URL_RE = re.compile(r"https?://|www\.")
SPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{10,}")
CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def clean_text(text: str, min_length: int = 80, max_urls: int = 5) -> str | None:
    """Clean single text line"""
    if not text or not isinstance(text, str):
        return None
    
    # Remove control chars
    text = CONTROL_RE.sub(" ", text)
    
    # Normalize whitespace
    text = SPACE_RE.sub(" ", text).strip()
    
    # Length check
    if len(text) < min_length:
        return None
    
    lower = text.lower()
    
    # Code detection (preserve code)
    is_code = any(p in text for p in CODE_PATTERNS)
    
    if not is_code:
        for p in BAD_PATTERNS:
            if p in lower:
                return None
    
    # URL count
    if len(URL_RE.findall(text)) > max_urls:
        return None
    
    # Alphanumeric ratio
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    if alnum / max(len(text), 1) < 0.25:
        return None
    
    # Repetition check
    if REPEAT_RE.search(text):
        return None
    
    return text


def process_chunk(lines: list, min_length: int, max_urls: int) -> list:
    """
    Process a chunk of lines in a worker
    Returns list of cleaned texts
    """
    cleaned = []
    for line in lines:
        text = clean_text(line, min_length, max_urls)
        if text:
            cleaned.append(text)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean tokenizer corpus (high performance)")
    parser.add_argument("--input", type=str, default="data/raw/tokenizer_corpus3.txt")
    parser.add_argument("--output", type=str, default="data/raw/tokenizer_corpus_clean.txt")
    parser.add_argument("--min-length", type=int, default=80)
    parser.add_argument("--max-urls", type=int, default=5)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Keural Corpus Cleaner (High Performance)")
    logger.info(f"Input:   {input_path}")
    logger.info(f"Output:  {output_path}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"RAM:     480GB available")
    logger.info("=" * 60)
    
    # Statistics
    total = 0
    kept = 0
    duplicates = 0
    hash_resets = 0
    
    # Deduplication set (memory bounded)
    seen_hashes = set()
    
    # Create pool
    pool = mp.Pool(args.workers)
    
    # Partial function for workers
    worker_func = partial(process_chunk, min_length=args.min_length, max_urls=args.max_urls)
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8", buffering=WRITE_BUFFER) as fout:
        
        # Read and process in chunks
        chunk = []
        pbar = tqdm(desc="Cleaning", unit="lines")
        
        for line in fin:
            total += 1
            chunk.append(line)
            pbar.update(1)
            
            if len(chunk) >= CHUNK_SIZE * args.workers:
                # Split into sub-chunks for each worker
                sub_chunks = [
                    chunk[i:i + CHUNK_SIZE] 
                    for i in range(0, len(chunk), CHUNK_SIZE)
                ]
                
                # Process in parallel
                results = pool.map(worker_func, sub_chunks)
                
                # Flatten results
                for cleaned_list in results:
                    for text in cleaned_list:
                        # Deduplication
                        h = hashlib.blake2b(text[:1000].encode(), digest_size=8).digest()
                        
                        if h in seen_hashes:
                            duplicates += 1
                            continue
                        
                        seen_hashes.add(h)
                        
                        # Memory guard: reset if too large
                        if len(seen_hashes) > MAX_HASHES:
                            logger.warning(f"Hash limit reached ({MAX_HASHES:,}), resetting...")
                            seen_hashes.clear()
                            hash_resets += 1
                        
                        # Write
                        fout.write(text + "\n")
                        kept += 1
                
                chunk = []
                pbar.set_postfix({"kept": kept, "dupes": duplicates})
        
        # Process remaining
        if chunk:
            sub_chunks = [
                chunk[i:i + CHUNK_SIZE] 
                for i in range(0, len(chunk), CHUNK_SIZE)
            ]
            results = pool.map(worker_func, sub_chunks)
            
            for cleaned_list in results:
                for text in cleaned_list:
                    h = hashlib.blake2b(text[:1000].encode(), digest_size=8).digest()
                    if h in seen_hashes:
                        duplicates += 1
                        continue
                    seen_hashes.add(h)
                    fout.write(text + "\n")
                    kept += 1
        
        pbar.close()
    
    pool.close()
    pool.join()
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total lines:     {total:,}")
    logger.info(f"Kept:            {kept:,} ({kept/total:.2%})")
    logger.info(f"Duplicates:      {duplicates:,}")
    logger.info(f"Hash resets:     {hash_resets}")
    logger.info(f"Input size:      {input_path.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"Output size:     {output_path.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"Throughput:      {total / (datetime.now() - datetime.now()).total_seconds():.0f} lines/sec" if False else "Done")
    logger.info(f"Output file:     {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


# production grade cleaning dataset, before training tokenizer model