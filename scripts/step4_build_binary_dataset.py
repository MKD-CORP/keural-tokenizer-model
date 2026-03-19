#!/usr/bin/env python3
"""
Build Binary Dataset - Production
Tokenizes, packs, and converts to binary in one pass
Streams data to handle 50GB+ datasets with low memory
"""

import os
import sys
import json
import glob
import struct
import argparse
import logging
import hashlib
import time
from datetime import datetime
from typing import Iterator, List, Optional, Tuple
from dataclasses import dataclass, asdict
import mmap

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ========================= CONFIG =========================

@dataclass
class BuildConfig:
    # Input/Output
    input_dir: str = "data/raw_stage1"
    output_dir: str = "data/binary"
    output_prefix: str = "keural"
    
    # Tokenizer
    tokenizer_path: str = "tokenizer/keural_tokenizer.model"
    
    # Sequence
    seq_length: int = 4096  # Stage 1: 4K context
    target_seq_tokens: int = 4096
    
    # Packing
    shuffle_buffer: int = 100000  # Documents to shuffle
    pack_buffer_tokens: int = 5000  # Pack slightly longer, then truncate to 4096
    
    # Output
    sequences_per_shard: int = 100000  # ~1.6GB per shard
    dtype: str = "uint32"
    
    # Parallel
    num_workers: int = 4
    
    # Logging
    log_interval: int = 10000

# ========================= LOGGING =========================

def setup_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"build_binary_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("BuildBinary")

logger = setup_logging()

# ========================= TOKENIZER =========================

class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        self.vocab_size = self.sp.vocab_size()
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        
        assert self.vocab_size == 131072, f"Expected 131072, got {self.vocab_size}"
        logger.info(f"Tokenizer loaded: {self.vocab_size} vocab")
    
    def encode(self, text: str) -> List[int]:
        """Encode with BOS/EOS"""
        ids = self.sp.encode(text, out_type=int)
        return [self.bos_id] + ids + [self.eos_id]
    
    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

# ========================= BINARY WRITER =========================

class BinaryShardWriter:
    """Writes sequences to binary format with index"""
    
    HEADER_MAGIC = b"KEURAL\x00\x00"
    HEADER_VERSION = 1
    HEADER_SIZE = 32
    
    def __init__(self, output_base: str, seq_length: int, dtype: str = "uint32"):
        self.output_base = output_base
        self.seq_length = seq_length
        self.dtype = np.dtype(dtype)
        
        self.bin_path = output_base + ".bin"
        self.idx_path = output_base + ".idx"
        self.meta_path = output_base + ".meta"
        
        self.bin_file = None
        self.idx_file = None
        self.num_sequences = 0
        self.current_offset = self.HEADER_SIZE
    
    def __enter__(self):
        self.bin_file = open(self.bin_path, "wb")
        self.idx_file = open(self.idx_path, "wb")
        
        # Reserve header space
        self.bin_file.write(b'\x00' * self.HEADER_SIZE)
        
        # Index header
        self.idx_file.write(struct.pack("<I", 0))  # num_sequences (placeholder)
        self.idx_file.write(struct.pack("<I", self.seq_length))
        
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def write_sequence(self, tokens: List[int]):
        """Write one sequence"""
        # Pad/truncate
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length - 1] + [2]  # Keep EOS
        elif len(tokens) < self.seq_length:
            tokens = tokens + [0] * (self.seq_length - len(tokens))
        
        tokens = tokens[:self.seq_length]
        
        # Write to binary
        arr = np.array(tokens, dtype=self.dtype)
        self.bin_file.write(arr.tobytes())
        
        # Write to index
        self.idx_file.write(struct.pack("<Q", self.current_offset))
        self.idx_file.write(struct.pack("<I", self.seq_length))
        
        self.current_offset += self.seq_length * self.dtype.itemsize
        self.num_sequences += 1
    
    def close(self):
        if self.bin_file:
            # Write final header
            self.bin_file.seek(0)
            header = struct.pack(
                "<8sIQQQ",
                self.HEADER_MAGIC,
                self.HEADER_VERSION,
                self.num_sequences,
                self.seq_length,
                0
            )
            self.bin_file.write(header)
            self.bin_file.close()
        
        if self.idx_file:
            # Update index header
            self.idx_file.seek(0)
            self.idx_file.write(struct.pack("<I", self.num_sequences))
            self.idx_file.close()
        
        # Write metadata
        meta = {
            "num_sequences": self.num_sequences,
            "seq_length": self.seq_length,
            "dtype": str(self.dtype),
            "file_size_bytes": os.path.getsize(self.bin_path) if os.path.exists(self.bin_path) else 0,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Shard closed: {self.num_sequences:,} sequences")

# ========================= MAIN BUILDER =========================

class BinaryDatasetBuilder:
    def __init__(self, config: BuildConfig):
        self.config = config
        self.tokenizer = Tokenizer(config.tokenizer_path)
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.stats = {
            "documents_processed": 0,
            "documents_skipped": 0,
            "tokens_processed": 0,
            "sequences_written": 0,
            "padding_added": 0,
            "truncated": 0,
            "shards_created": 0,
        }
        
        self.rng = np.random.RandomState(42)
    
    def get_input_files(self) -> List[str]:
        """Get all input text files"""
        files = []
        for pattern in ["*.txt", "*.jsonl"]:
            files.extend(glob.glob(os.path.join(self.config.input_dir, pattern)))
        
        if not files:
            raise ValueError(f"No files found in {self.config.input_dir}")
        
        # Shuffle files for randomness
        self.rng.shuffle(files)
        logger.info(f"Found {len(files)} input files")
        return files
    
    def document_iterator(self, files: List[str]) -> Iterator[str]:
        """Stream documents with large shuffle buffer"""
        buffer = []
        
        for file_path in files:
            logger.info(f"Reading: {os.path.basename(file_path)}")
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    buffer.append(line)
                    
                    if len(buffer) >= self.config.shuffle_buffer:
                        self.rng.shuffle(buffer)
                        while buffer:
                            yield buffer.pop(0)
        
        # Flush remaining
        self.rng.shuffle(buffer)
        while buffer:
            yield buffer.pop(0)
    
    def build(self):
        """Main build process"""
        logger.info("=" * 70)
        logger.info("BUILDING BINARY DATASET")
        logger.info("=" * 70)
        logger.info(f"Input: {self.config.input_dir}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"Sequence length: {self.config.seq_length}")
        logger.info(f"Target per shard: {self.config.sequences_per_shard:,} sequences")
        
        files = self.get_input_files()
        
        # Process
        shard_idx = 0
        sequences_in_shard = 0
        
        output_base = os.path.join(
            self.config.output_dir,
            f"{self.config.output_prefix}_{shard_idx:03d}"
        )
        
        writer = BinaryShardWriter(output_base, self.config.seq_length, self.config.dtype)
        writer.__enter__()
        
        # Packing buffer
        pack_buffer = []
        pack_buffer_len = 0
        
        pbar = tqdm(desc="Processing documents", unit="docs")
        
        try:
            for doc_text in self.document_iterator(files):
                # Tokenize
                try:
                    tokens = self.tokenizer.encode(doc_text)
                except Exception as e:
                    logger.warning(f"Tokenization failed: {e}")
                    self.stats["documents_skipped"] += 1
                    continue
                
                doc_len = len(tokens)
                
                # Skip very long single documents
                if doc_len > self.config.pack_buffer_tokens:
                    tokens = tokens[:self.config.pack_buffer_tokens - 1] + [2]
                    doc_len = len(tokens)
                    self.stats["truncated"] += 1
                
                # Check if fits in pack buffer
                if pack_buffer_len + doc_len > self.config.pack_buffer_tokens and pack_buffer:
                    # Flush pack buffer to sequence
                    seq_tokens = []
                    for t in pack_buffer:
                        seq_tokens.extend(t)
                    
                    # Truncate to exact length
                    if len(seq_tokens) > self.config.seq_length:
                        seq_tokens = seq_tokens[:self.config.seq_length - 1] + [2]
                        self.stats["truncated"] += 1
                    
                    # Write
                    writer.write_sequence(seq_tokens)
                    self.stats["sequences_written"] += 1
                    self.stats["padding_added"] += max(0, self.config.seq_length - len(seq_tokens))
                    sequences_in_shard += 1
                    
                    # Check shard rotation
                    if sequences_in_shard >= self.config.sequences_per_shard:
                        writer.close()
                        self.stats["shards_created"] += 1
                        
                        shard_idx += 1
                        sequences_in_shard = 0
                        
                        output_base = os.path.join(
                            self.config.output_dir,
                            f"{self.config.output_prefix}_{shard_idx:03d}"
                        )
                        writer = BinaryShardWriter(output_base, self.config.seq_length, self.config.dtype)
                        writer.__enter__()
                        
                        logger.info(f"Rotated to shard {shard_idx}")
                    
                    # Reset pack buffer
                    pack_buffer = [tokens]
                    pack_buffer_len = doc_len
                else:
                    # Add to pack buffer
                    pack_buffer.append(tokens)
                    pack_buffer_len += doc_len
                
                self.stats["documents_processed"] += 1
                self.stats["tokens_processed"] += doc_len - 2  # Exclude BOS/EOS
                
                pbar.update(1)
                
                if self.stats["documents_processed"] % self.config.log_interval == 0:
                    self._log_progress()
            
            # Flush final pack buffer
            if pack_buffer:
                seq_tokens = []
                for t in pack_buffer:
                    seq_tokens.extend(t)
                
                if len(seq_tokens) > self.config.seq_length:
                    seq_tokens = seq_tokens[:self.config.seq_length - 1] + [2]
                
                writer.write_sequence(seq_tokens)
                self.stats["sequences_written"] += 1
                self.stats["padding_added"] += max(0, self.config.seq_length - len(seq_tokens))
            
            pbar.close()
            
        finally:
            writer.close()
            self.stats["shards_created"] += 1
        
        self._save_stats()
        self._print_final_stats()
    
    def _log_progress(self):
        """Log current progress"""
        docs = self.stats["documents_processed"]
        seqs = self.stats["sequences_written"]
        tokens = self.stats["tokens_processed"]
        
        logger.info(
            f"Progress: {docs:,} docs, {seqs:,} seqs, "
            f"{tokens:,} tokens ({tokens/1e9:.2f}B)"
        )
    
    def _save_stats(self):
        """Save build statistics"""
        stats_file = os.path.join(self.config.output_dir, "build_stats.json")
        
        # Calculate utilization
        total_positions = self.stats["sequences_written"] * self.config.seq_length
        utilization = 100 * (1 - self.stats["padding_added"] / total_positions) if total_positions > 0 else 0
        
        final_stats = {
            "config": asdict(self.config),
            "statistics": self.stats,
            "calculated": {
                "total_tokens_billions": self.stats["tokens_processed"] / 1e9,
                "sequences_millions": self.stats["sequences_written"] / 1e6,
                "avg_tokens_per_seq": self.stats["tokens_processed"] / max(1, self.stats["sequences_written"]),
                "sequence_utilization_percent": utilization,
                "estimated_training_hours_a100": (self.stats["tokens_processed"] / 200000) / 3600,
            }
        }
        
        with open(stats_file, "w") as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f"Stats saved: {stats_file}")
    
    def _print_final_stats(self):
        """Print final statistics"""
        logger.info("=" * 70)
        logger.info("BUILD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Documents processed: {self.stats['documents_processed']:,}")
        logger.info(f"Documents skipped:   {self.stats['documents_skipped']:,}")
        logger.info(f"Tokens processed:    {self.stats['tokens_processed']:,} ({self.stats['tokens_processed']/1e9:.2f}B)")
        logger.info(f"Sequences written:   {self.stats['sequences_written']:,}")
        logger.info(f"Shards created:      {self.stats['shards_created']}")
        logger.info(f"Padding tokens:      {self.stats['padding_added']:,}")
        
        total_pos = self.stats["sequences_written"] * self.config.seq_length
        if total_pos > 0:
            util = 100 * (1 - self.stats["padding_added"] / total_pos)
            logger.info(f"Sequence utilization: {util:.2f}%")
        
        # Output files
        bins = glob.glob(os.path.join(self.config.output_dir, "*.bin"))
        total_size = sum(os.path.getsize(f) for f in bins) / 1024**3
        logger.info(f"Total binary size: {total_size:.2f} GB")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 70)

# ========================= MAIN =========================

def main():
    parser = argparse.ArgumentParser(description="Build binary dataset for training")
    parser.add_argument("--input_dir", default="data/raw_stage1")
    parser.add_argument("--output_dir", default="data/binary")
    parser.add_argument("--output_prefix", default="keural")
    parser.add_argument("--tokenizer", default="tokenizer/keural_tokenizer.model")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--sequences_per_shard", type=int, default=100000)
    parser.add_argument("--shuffle_buffer", type=int, default=100000)
    
    args = parser.parse_args()
    
    config = BuildConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        tokenizer_path=args.tokenizer,
        seq_length=args.seq_length,
        sequences_per_shard=args.sequences_per_shard,
        shuffle_buffer=args.shuffle_buffer,
    )
    
    builder = BinaryDatasetBuilder(config)
    builder.build()

if __name__ == "__main__":
    main()


    # Tokenization → Sequence Packing → Binary Dataset Creation