#!/usr/bin/env python3
"""
Keural Tokenizer Corpus Builder - Production Ready
Fixed streaming dataset loading
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"data/logs/corpus_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    config: Optional[str]
    split: str
    target: int
    text_field: str
    lang_filter: Optional[str] = None


# CORRECTED dataset configurations - ALL USE STREAMING
DATASETS = {
    "fineweb_en": DatasetConfig(
        "HuggingFaceFW/fineweb",
        "default",
        "train",
        1_000_000,
        "text",
        "en"
    ),
    "fineweb2_ko": DatasetConfig(
        "HuggingFaceFW/fineweb-2",
        "kor_Hang",
        "train",
        500_000,
        "text",
        "ko"
    ),
    "dolma": DatasetConfig(
        "allenai/dolma",
        "v1_6",
        "train",
        600_000,
        "text"
    ),
    "c4_en": DatasetConfig(
        "allenai/c4",
        "en",
        "train",
        600_000,
        "text",
        "en"
    ),
    "c4_ko": DatasetConfig(
        "allenai/c4",
        "ko",
        "train",
        600_000,
        "text",
        "ko"
    ),
    "cc100_ko": DatasetConfig(
        "cc100",
        "ko",
        "train",
        300_000,
        "text",
        "ko"
    ),
    "stack_python": DatasetConfig(
        "bigcode/the-stack-v2",
        "Python",
        "train",
        400_000,
        "content"
    ),
    "starcoder": DatasetConfig(
        "bigcode/starcoderdata",
        None,
        "train",
        200_000,
        "content"
    ),
    "wiki_en": DatasetConfig(
        "wikimedia/wikipedia",
        "20231101.en",
        "train",
        300_000,
        "text"
    ),
    "wiki_ko": DatasetConfig(
        "wikimedia/wikipedia",
        "20231101.ko",
        "train",
        150_000,
        "text"
    ),
    "arxiv": DatasetConfig(
        "scientific_papers",
        "arxiv",
        "train",
        300_000,
        "article"
    ),
    "pmc": DatasetConfig(
        "scientific_papers",
        "pubmed",
        "train",
        200_000,
        "article"
    ),
    "openwebmath": DatasetConfig(
        "open-web-math/open-web-math",
        None,
        "train",
        200_000,
        "text"
    ),
    "gutenberg": DatasetConfig(
        "bookcorpus",
        None,
        "train",
        250_000,
        "text"
    ),
}


class CorpusBuilder:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.raw_dir = self.base_dir / "data" / "raw"
        self.logs_dir = self.base_dir / "data" / "logs"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_file = self.raw_dir / "tokenizer_corpus.txt"
        self.progress_file = self.logs_dir / "tokenizer_progress.json"
        
        self.progress = self._load_progress()
        
    def _load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {k: 0 for k in DATASETS}
    
    def _save_progress(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)
    
    def _clean_text(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        text = text.replace("\n", " ").replace("\r", " ").strip()
        return text if len(text) >= 50 else None
    
    def _load_dataset_safe(self, config: DatasetConfig):
        """Load dataset with proper streaming handling"""
        try:
            logger.info(f"  Loading {config.name}...")
            
            # ALWAYS use streaming for large datasets to avoid memory issues
            ds = load_dataset(
                config.name,
                config.config,
                split=config.split,
                streaming=True,
                trust_remote_code=True
            )
            return ds
                
        except Exception as e:
            logger.error(f"  Failed to load {config.name}: {e}")
            raise
    
    def process_dataset(self, key: str, config: DatasetConfig):
        target = config.target
        collected = self.progress.get(key, 0)
        
        if collected >= target:
            logger.info(f"{key}: Already complete ({collected:,}/{target:,})")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {key}")
        logger.info(f"Progress: {collected:,} / {target:,}")
        logger.info(f"Dataset: {config.name} | Config: {config.config}")
        
        try:
            dataset = self._load_dataset_safe(config)
        except Exception as e:
            logger.error(f"Skipping {key} due to load error")
            return
        
        count = collected
        written_this_session = 0
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            pbar = tqdm(
                total=target,
                initial=collected,
                desc=key[:20],
                unit="docs",
                ncols=80,
                mininterval=5.0  # Update every 5 seconds to reduce overhead
            )
            
            try:
                for item in dataset:
                    if count >= target:
                        break
                    
                    # Skip already processed items (for resume)
                    if count < collected:
                        count += 1
                        continue
                    
                    # Extract text
                    text = item.get(config.text_field)
                    text = self._clean_text(text)
                    if not text:
                        continue
                    
                    # Write
                    f.write(text + "\n")
                    f.flush()  # Ensure write to disk
                    count += 1
                    written_this_session += 1
                    self.progress[key] = count
                    pbar.update(1)
                    
                    # Save progress periodically
                    if written_this_session % 10000 == 0:
                        self._save_progress()
                        
            except KeyboardInterrupt:
                logger.warning("\nInterrupted by user")
                raise
            except Exception as e:
                logger.error(f"Error during processing: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
            finally:
                pbar.close()
                self._save_progress()
        
        logger.info(f"Completed {key}: {count:,} total samples")
    
    def build(self):
        logger.info("="*60)
        logger.info("Keural Tokenizer Corpus Builder")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Total target: {sum(d.target for d in DATASETS.values()):,} samples")
        logger.info("="*60)
        
        # Clear output file if starting fresh
        if sum(self.progress.values()) == 0 and self.output_file.exists():
            logger.info("Starting fresh - clearing existing corpus file")
            self.output_file.unlink()
            # Create empty file
            self.output_file.touch()
        
        for key, config in DATASETS.items():
            try:
                self.process_dataset(key, config)
            except Exception as e:
                logger.error(f"Fatal error in {key}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info("\n" + "="*60)
        logger.info("BUILD COMPLETE")
        total = sum(self.progress.values())
        target_total = sum(d.target for d in DATASETS.values())
        logger.info(f"Total collected: {total:,} / {target_total:,}")
        logger.info(f"Output: {self.output_file}")
        if self.output_file.exists():
            logger.info(f"Size: {self.output_file.stat().st_size / (1024**3):.2f} GB")
        logger.info("="*60)


def main():
    builder = CorpusBuilder()
    builder.build()


if __name__ == "__main__":
    main()


# this script for collecting corpus for tokenizer model trainning