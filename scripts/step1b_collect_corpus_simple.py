#!/usr/bin/env python3

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_FILE = "tokenizer_corpus.txt"
TARGET_DOCS = 5_300_000

datasets_plan = [

    # -------- General Web --------
    ("HuggingFaceFW/fineweb", "train", "text", 1_000_000),
    ("HuggingFaceFW/fineweb", "train", "text", 500_000),

    ("allenai/dolma", "train", "text", 600_000),

    ("allenai/c4", "en", "text", 600_000),
    ("allenai/c4", "ko", "text", 600_000),

    ("cc100", "ko", "text", 300_000),

    # -------- Code --------
    ("bigcode/the-stack-v2", "python", "content", 400_000),
    ("bigcode/starcoderdata", "train", "content", 200_000),

    # -------- Wikipedia --------
    ("wikipedia", "20220301.en", "text", 300_000),
    ("wikipedia", "20220301.ko", "text", 150_000),

    # -------- Scientific --------
    ("arxiv_dataset", "default", "abstract", 300_000),
    ("pubmed", "train", "abstract", 200_000),

    # -------- Math --------
    ("openwebmath", "train", "text", 200_000),

    # -------- Books --------
    ("pg19", "train", "text", 250_000),
]


def clean_text(t):

    if not t:
        return None

    t = t.replace("\x00", " ")
    t = t.replace("\r", " ")
    t = t.replace("\t", " ")
    t = t.strip()

    if len(t) < 50:
        return None

    return t


def main():

    os.makedirs("data", exist_ok=True)

    total_written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for name, subset, field, target in datasets_plan:

            print(f"\nCollecting {name} / {subset}")

            ds = load_dataset(
                name,
                subset,
                split="train",
                streaming=True
            )

            count = 0

            for sample in tqdm(ds, total=target):

                text = sample.get(field)

                text = clean_text(text)

                if text is None:
                    continue

                out.write(text + "\n")

                count += 1
                total_written += 1

                if count >= target:
                    break

            print(f"Collected {count} documents")

    print("\nCorpus complete")
    print("Documents:", total_written)


if __name__ == "__main__":
    main()