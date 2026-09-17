#!/usr/bin/env python3
"""Deterministic benchmark corpus and gold dataset generator for DocuLens AI."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.corpus_builder import build_corpus_pdfs
from scripts.generate_gold_queries import write_gold_dataset


def main() -> None:
    samples_dir = PROJECT_ROOT / "data" / "samples"
    dataset_file = PROJECT_ROOT / "data" / "gold_dataset.jsonl"

    print("🚀 Generating DocuLens AI 10-Document Canonical Evaluation Corpus...")
    created_pdfs = build_corpus_pdfs(samples_dir)
    print(f"✅ Generated {len(created_pdfs)} PDF sample documents in {samples_dir}")

    print("🚀 Compiling DocuLens AI Gold Benchmark Dataset...")
    written_dataset = write_gold_dataset(dataset_file)
    print(f"✅ Written gold dataset to {written_dataset}")


if __name__ == "__main__":
    main()
