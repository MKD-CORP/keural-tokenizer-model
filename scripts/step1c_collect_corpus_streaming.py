#!/usr/bin/env python3
"""
Keural Tokenizer Corpus Builder - Production Ready

What this script does:
- Streams large datasets safely
- Builds a multilingual tokenizer corpus
- Writes one output file per dataset part
- Supports resume through a JSON progress file
- Applies light text cleaning and language filtering
- Merges all parts into one final tokenizer corpus file

Recommended use:
    python scripts/build_tokenizer_corpus.py
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterable

from datasets import load_dataset
from tqdm import tqdm


# =========================
# Configuration
# =========================

TARGET_TOTAL_SIZE_GB = 54.77
TARGET_TOTAL_SIZE_BYTES = int(TARGET_TOTAL_SIZE_GB * 1024**3)

LOG_SAVE_EVERY = 10_000
FLUSH_EVERY = 2_000
MIN_TEXT_LEN = 80

# If True, stop once merged corpus reaches TARGET_TOTAL_SIZE_GB
ENFORCE_TOTAL_SIZE_CAP = False

# If True, create final merged tokenizer_corpus.txt after all parts are done
MERGE_AT_END = True


@dataclass
class DatasetSource:
    name: str
    config: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    lang_filter: Optional[str] = None


@dataclass
class DatasetPlan:
    key: str
    target_docs: int
    sources: List[DatasetSource]


# =========================
# Dataset Plan
# =========================
# Notes:
# - Primary + fallback IDs are provided where possible.
# - Only one source is used per dataset key: first one that loads successfully.
# - Some ecosystem dataset IDs changed over time, so fallbacks are important.

DATASET_PLANS: List[DatasetPlan] = [
    DatasetPlan(
        key="fineweb_en",
        target_docs=1_000_000,
        sources=[
            DatasetSource("HuggingFaceFW/fineweb", "default", "train", "text", "en"),
            DatasetSource("HuggingFaceFW/fineweb", None, "train", "text", "en"),
        ],
    ),
    DatasetPlan(
        key="fineweb2_ko",
        target_docs=500_000,
        sources=[
            DatasetSource("HuggingFaceFW/fineweb-2", "kor_Hang", "train", "text", "ko"),
            DatasetSource("HuggingFaceFW/fineweb-2", "ko_Kore", "train", "text", "ko"),
            DatasetSource("mc4", "ko", "train", "text", "ko"),  # fallback if fineweb-2 config changes
        ],
    ),
    DatasetPlan(
        key="dolma",
        target_docs=600_000,
        sources=[
            DatasetSource("allenai/dolma", "v1_7", "train", "text", None),
            DatasetSource("allenai/dolma", "v1_6", "train", "text", None),
        ],
    ),
    DatasetPlan(
        key="c4_en",
        target_docs=600_000,
        sources=[
            DatasetSource("allenai/c4", "en", "train", "text", "en"),
        ],
    ),
    DatasetPlan(
        key="c4_ko",
        target_docs=600_000,
        sources=[
            DatasetSource("allenai/c4", "ko", "train", "text", "ko"),
            DatasetSource("mc4", "ko", "train", "text", "ko"),  # fallback
        ],
    ),
    DatasetPlan(
        key="cc100_ko",
        target_docs=300_000,
        sources=[
            DatasetSource("cc100", "ko", "train", "text", "ko"),
            DatasetSource("mc4", "ko", "train", "text", "ko"),
        ],
    ),
    DatasetPlan(
        key="stack_code",
        target_docs=400_000,
        sources=[
            DatasetSource("bigcode/the-stack", None, "train", "content", None),
            DatasetSource("bigcode/the-stack-v2", None, "train", "content", None),
        ],
    ),
    DatasetPlan(
        key="starcoder",
        target_docs=200_000,
        sources=[
            DatasetSource("bigcode/starcoderdata", None, "train", "content", None),
        ],
    ),
    DatasetPlan(
        key="wiki_en",
        target_docs=300_000,
        sources=[
            DatasetSource("wikimedia/wikipedia", "20231101.en", "train", "text", "en"),
            DatasetSource("wikipedia", "20220301.en", "train", "text", "en"),
        ],
    ),
    DatasetPlan(
        key="wiki_ko",
        target_docs=150_000,
        sources=[
            DatasetSource("wikimedia/wikipedia", "20231101.ko", "train", "text", "ko"),
            DatasetSource("wikipedia", "20220301.ko", "train", "text", "ko"),
        ],
    ),
    DatasetPlan(
        key="arxiv",
        target_docs=300_000,
        sources=[
            DatasetSource("scientific_papers", "arxiv", "train", "article", None),
        ],
    ),
    DatasetPlan(
        key="pubmed",
        target_docs=200_000,
        sources=[
            DatasetSource("scientific_papers", "pubmed", "train", "article", None),
        ],
    ),
    DatasetPlan(
        key="openwebmath",
        target_docs=200_000,
        sources=[
            DatasetSource("open-web-math/open-web-math", None, "train", "text", None),
            DatasetSource("openwebmath", None, "train", "text", None),
        ],
    ),
    DatasetPlan(
        key="gutenberg",
        target_docs=250_000,
        sources=[
            DatasetSource("pg19", None, "train", "text", None),
        ],
    ),
]


# =========================
# Logging
# =========================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"corpus_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("keural_corpus_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =========================
# Builder
# =========================

class CorpusBuilder:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.logs_dir = self.data_dir / "logs"
        self.parts_dir = self.raw_dir / "tokenizer_parts"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.parts_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.raw_dir / "tokenizer_corpus3.txt"
        self.progress_file = self.logs_dir / "tokenizer_progress3.json"

        self.logger = setup_logging(self.logs_dir)
        self.progress = self._load_progress()

    # ---------- progress ----------

    def _default_progress(self) -> Dict[str, Any]:
        return {
            "datasets": {
                plan.key: {
                    "accepted_docs": 0,
                    "selected_source": None,
                    "part_file": str(self.parts_dir / f"{plan.key}.txt"),
                    "done": False,
                    "last_update": None,
                }
                for plan in DATASET_PLANS
            },
            "meta": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
        }

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        progress = self._default_progress()
        self._save_progress(progress)
        return progress

    def _save_progress(self, progress: Optional[Dict[str, Any]] = None) -> None:
        if progress is None:
            progress = self.progress
        progress["meta"]["updated_at"] = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    # ---------- helpers ----------

    def _get_plan(self, key: str) -> DatasetPlan:
        for plan in DATASET_PLANS:
            if plan.key == key:
                return plan
        raise KeyError(f"Unknown dataset key: {key}")

    def _part_file(self, key: str) -> Path:
        return self.parts_dir / f"{key}.txt"

    def _current_total_size_bytes(self) -> int:
        total = 0
        for plan in DATASET_PLANS:
            p = self._part_file(plan.key)
            if p.exists():
                total += p.stat().st_size
        return total

    def _extract_text(self, item: Dict[str, Any], text_field: str) -> Optional[str]:
        value = item.get(text_field)

        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts = [x for x in value if isinstance(x, str)]
            return "\n".join(parts).strip() if parts else None

        return None

    def _light_clean(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None

        text = text.replace("\x00", " ")
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < MIN_TEXT_LEN:
            return None

        # obvious boilerplate / junk
        lower = text.lower()
        bad_snippets = [
            "cookie policy",
            "privacy policy",
            "all rights reserved",
            "javascript is disabled",
            "enable javascript",
            "sign up for our newsletter",
        ]
        if any(x in lower for x in bad_snippets):
            return None

        # too many URLs => likely junk / navigation / link dump
        if len(re.findall(r"https?://|www\.", lower)) > 5:
            return None

        # too many repeated punctuation blocks
        if text.count("|||") > 2 or text.count("***") > 4:
            return None

        # ultra-low alnum content => likely garbage
        alnum = sum(ch.isalnum() for ch in text)
        if alnum / max(len(text), 1) < 0.25:
            return None

        return text

    def _lang_ok(self, text: str, lang: Optional[str]) -> bool:
        if not lang:
            return True

        if not text:
            return False

        if lang == "en":
            ascii_letters = sum((ord(c) < 128 and c.isalpha()) for c in text)
            ascii_ratio = ascii_letters / max(len(text), 1)
            return ascii_ratio >= 0.20

        if lang == "ko":
            hangul = sum('\uac00' <= c <= '\ud7a3' for c in text)
            return hangul >= 10

        return True

    def _load_stream(self, source: DatasetSource):
        kwargs = {
            "path": source.name,
            "split": source.split,
            "streaming": True,
            "trust_remote_code": True,
        }
        if source.config is not None:
            kwargs["name"] = source.config
        return load_dataset(**kwargs)

    def _resolve_source(self, key: str, plan: DatasetPlan) -> DatasetSource:
        saved = self.progress["datasets"][key]["selected_source"]
        if saved:
            for s in plan.sources:
                if s.name == saved["name"] and s.config == saved["config"]:
                    self.logger.info(f"{key}: reusing saved source {s.name} | config={s.config}")
                    return s

        last_error = None
        for source in plan.sources:
            try:
                self.logger.info(f"{key}: trying source {source.name} | config={source.config}")
                _ = self._load_stream(source)
                self.progress["datasets"][key]["selected_source"] = asdict(source)
                self._save_progress()
                self.logger.info(f"{key}: selected source {source.name} | config={source.config}")
                return source
            except Exception as e:
                last_error = e
                self.logger.warning(f"{key}: failed source {source.name} | config={source.config} | error={e}")

        raise RuntimeError(f"{key}: no working dataset source found. Last error: {last_error}")

    def _skip_already_accepted(self, dataset: Iterable[Dict[str, Any]], source: DatasetSource, accepted_docs: int) -> Iterable[Dict[str, Any]]:
        """
        Resume logic for streaming datasets:
        We must consume the stream and count accepted docs using the same filters.
        This is slower on resume, but it is correct.
        """
        if accepted_docs <= 0:
            return dataset

        self.logger.info(f"Resuming stream by skipping {accepted_docs:,} already accepted docs...")
        skipped = 0
        pbar = tqdm(total=accepted_docs, desc="resume-skip", unit="docs", ncols=80)

        iterator = iter(dataset)
        while skipped < accepted_docs:
            item = next(iterator)
            text = self._extract_text(item, source.text_field)
            if text is None:
                continue
            text = self._light_clean(text)
            if text is None:
                continue
            if not self._lang_ok(text, source.lang_filter):
                continue
            skipped += 1
            pbar.update(1)

        pbar.close()

        def remaining():
            for item in iterator:
                yield item

        return remaining()

    def _merge_parts(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("Merging tokenizer part files...")
        if self.output_file.exists():
            self.output_file.unlink()

        total_bytes = 0
        with open(self.output_file, "wb") as out:
            for plan in DATASET_PLANS:
                part = self._part_file(plan.key)
                if not part.exists():
                    continue

                self.logger.info(f"Merging {part.name} ...")
                with open(part, "rb") as src:
                    while True:
                        chunk = src.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        total_bytes += len(chunk)

        self.logger.info(f"Merged output: {self.output_file}")
        self.logger.info(f"Merged size: {total_bytes / (1024**3):.2f} GB")
        self.logger.info("=" * 70)

    # ---------- main processing ----------

    def process_plan(self, plan: DatasetPlan) -> None:
        key = plan.key
        state = self.progress["datasets"][key]
        accepted_docs = int(state["accepted_docs"])
        target_docs = int(plan.target_docs)
        part_file = self._part_file(key)

        if state["done"] or accepted_docs >= target_docs:
            self.logger.info(f"{key}: already complete ({accepted_docs:,}/{target_docs:,})")
            state["done"] = True
            self._save_progress()
            return

        if ENFORCE_TOTAL_SIZE_CAP and self._current_total_size_bytes() >= TARGET_TOTAL_SIZE_BYTES:
            self.logger.warning("Global size cap already reached. Stopping further collection.")
            return

        source = self._resolve_source(key, plan)

        self.logger.info("=" * 70)
        self.logger.info(f"Processing dataset: {key}")
        self.logger.info(f"Target docs: {target_docs:,}")
        self.logger.info(f"Accepted docs so far: {accepted_docs:,}")
        self.logger.info(f"Source: {source.name} | config={source.config} | field={source.text_field}")
        self.logger.info(f"Part file: {part_file}")
        self.logger.info("=" * 70)

        dataset = self._load_stream(source)
        dataset = self._skip_already_accepted(dataset, source, accepted_docs)

        written_this_session = 0
        bytes_written_this_session = 0
        start_time = time.time()

        with open(part_file, "a", encoding="utf-8") as f:
            pbar = tqdm(
                total=target_docs,
                initial=accepted_docs,
                desc=key,
                unit="docs",
                ncols=100,
                mininterval=5.0,
            )

            try:
                for item in dataset:
                    if accepted_docs >= target_docs:
                        break

                    if ENFORCE_TOTAL_SIZE_CAP and self._current_total_size_bytes() >= TARGET_TOTAL_SIZE_BYTES:
                        self.logger.warning("Global size cap reached during collection. Stopping.")
                        break

                    text = self._extract_text(item, source.text_field)
                    if text is None:
                        continue

                    text = self._light_clean(text)
                    if text is None:
                        continue

                    if not self._lang_ok(text, source.lang_filter):
                        continue

                    f.write(text + "\n")
                    accepted_docs += 1
                    written_this_session += 1
                    bytes_written_this_session += len((text + "\n").encode("utf-8"))

                    state["accepted_docs"] = accepted_docs
                    state["last_update"] = datetime.now().isoformat()

                    pbar.update(1)

                    if written_this_session % FLUSH_EVERY == 0:
                        f.flush()

                    if written_this_session % LOG_SAVE_EVERY == 0:
                        self._save_progress()
                        elapsed = max(time.time() - start_time, 1e-6)
                        speed = written_this_session / elapsed
                        self.logger.info(
                            f"{key}: session_docs={written_this_session:,} | "
                            f"total_docs={accepted_docs:,}/{target_docs:,} | "
                            f"speed={speed:.2f} docs/s | "
                            f"session_size={bytes_written_this_session / (1024**3):.2f} GB"
                        )

            except KeyboardInterrupt:
                self.logger.warning(f"{key}: interrupted by user")
                raise
            except Exception as e:
                self.logger.exception(f"{key}: runtime error: {e}")
            finally:
                f.flush()
                pbar.close()
                state["accepted_docs"] = accepted_docs
                state["done"] = accepted_docs >= target_docs
                state["last_update"] = datetime.now().isoformat()
                self._save_progress()

        elapsed = max(time.time() - start_time, 1e-6)
        self.logger.info(
            f"{key}: finished session | accepted={accepted_docs:,}/{target_docs:,} | "
            f"part_size={part_file.stat().st_size / (1024**3):.2f} GB | "
            f"avg_speed={written_this_session / elapsed:.2f} docs/s"
        )

    def build(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("Keural Tokenizer Corpus Builder - Production Ready")
        self.logger.info(f"Base dir: {self.base_dir}")
        self.logger.info(f"Parts dir: {self.parts_dir}")
        self.logger.info(f"Final output: {self.output_file}")
        self.logger.info(f"Target total docs: {sum(p.target_docs for p in DATASET_PLANS):,}")
        self.logger.info(f"Target total size: {TARGET_TOTAL_SIZE_GB:.2f} GB")
        self.logger.info("=" * 70)

        for plan in DATASET_PLANS:
            self.process_plan(plan)

        if MERGE_AT_END:
            self._merge_parts()

        total_docs = sum(self.progress["datasets"][p.key]["accepted_docs"] for p in DATASET_PLANS)
        total_bytes = self._current_total_size_bytes()

        self.logger.info("=" * 70)
        self.logger.info("BUILD COMPLETE")
        self.logger.info(f"Collected docs: {total_docs:,}")
        self.logger.info(f"Parts total size: {total_bytes / (1024**3):.2f} GB")
        if self.output_file.exists():
            self.logger.info(f"Final merged corpus: {self.output_file}")
            self.logger.info(f"Final merged size: {self.output_file.stat().st_size / (1024**3):.2f} GB")
        self.logger.info("=" * 70)


def main() -> None:
    builder = CorpusBuilder()
    builder.build()


if __name__ == "__main__":
    main()



# production grade script