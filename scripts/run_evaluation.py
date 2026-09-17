#!/usr/bin/env python3
"""Automated Evaluation and Regression CLI for DocuLens AI — Phase 8.7.

Usage examples:
    python scripts/run_evaluation.py --config all
    python scripts/run_evaluation.py --config dense --save-baseline data/baseline_dense.json
    python scripts/run_evaluation.py --config dense --baseline data/baseline_dense.json --strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.harness import (
    BASELINE_NAMES,
    EvaluationStores,
    create_isolated_stores,
    get_shared_stores,
    index_evaluation_corpus,
    index_evaluation_document,
    run_all_baselines,
    run_baseline_evaluation,
)
from app.evaluation.regression import RegressionThresholds, compare_against_baseline
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExperimentResult,
    ExperimentSummary,
    RegressionReport,
)
from app.services.dataset import load_gold_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DocuLens AI Evaluation & Regression Automation Harness"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "data/gold_dataset.jsonl"),
        help="Path to gold dataset JSONL file",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=str(PROJECT_ROOT / "data/samples"),
        help="Path to directory containing all evaluation corpus PDFs (indexes all *.pdf in directory)",
    )
    parser.add_argument(
        "--corpus-pdf",
        type=str,
        default=str(PROJECT_ROOT / "data/samples/doculens-architecture-v1.pdf"),
        help="Path to canonical evaluation corpus PDF",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["all", *BASELINE_NAMES],
        help="Named baseline configuration to evaluate or 'all'",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default="retrieval",
        choices=["retrieval", "generation", "end_to_end"],
        help="Evaluation boundary: 'retrieval' (offline/no API keys), 'generation', or 'end_to_end'",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for Recall@K and MRR@K",
    )
    parser.add_argument(
        "--default-k",
        type=int,
        default=5,
        help="Default K value for summary metrics",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to prior baseline JSON result for regression detection",
    )
    parser.add_argument(
        "--save-baseline",
        type=str,
        default=None,
        help="Path to save current evaluation results as a baseline JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save machine-readable JSON evaluation results",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        help="Path to save human-readable Markdown evaluation report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any baseline fails quality thresholds or detects regression",
    )
    return parser.parse_args()


def format_summary_table(results: dict[str, ExperimentResult]) -> str:
    """Format evaluation experiment results into an ASCII table."""
    lines = [
        "",
        "=" * 100,
        f"{'CONFIGURATION':<22} | {'RECALL@5':<9} | {'MRR@5':<8} | {'CTX PREC':<9} | {'CTX REC':<8} | {'LATENCY (p95)':<14} | {'GATE':<6}",
        "-" * 100,
    ]
    for name, res in results.items():
        s = res.summary
        rec = f"{s.mean_recall_at_k:.4f}"
        mrr = f"{s.mean_mrr_at_k:.4f}"
        cp = f"{s.mean_context_precision:.4f}" if s.mean_context_precision is not None else "N/A"
        cr = f"{s.mean_context_recall:.4f}" if s.mean_context_recall is not None else "N/A"
        lat = f"{s.p95_total_latency_ms:.1f}ms"
        status = "PASS" if s.thresholds_passed else "FAIL"
        lines.append(f"{name:<22} | {rec:<9} | {mrr:<8} | {cp:<9} | {cr:<8} | {lat:<14} | {status:<6}")
    lines.append("=" * 100)
    lines.append("")
    return "\n".join(lines)


def format_markdown_report(
    results: dict[str, ExperimentResult],
    regression_reports: dict[str, RegressionReport] | None = None,
) -> str:
    """Generate a structured Markdown evaluation report."""
    md = [
        "# DocuLens AI — Automated Evaluation Report",
        "",
        "## 1. Summary of Baseline Configurations",
        "",
        "| Configuration | Recall@5 | MRR@5 | Context Precision | Context Recall | Latency (p95) | Gate |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, res in results.items():
        s = res.summary
        rec = f"{s.mean_recall_at_k:.4f}"
        mrr = f"{s.mean_mrr_at_k:.4f}"
        cp = f"{s.mean_context_precision:.4f}" if s.mean_context_precision is not None else "N/A"
        cr = f"{s.mean_context_recall:.4f}" if s.mean_context_recall is not None else "N/A"
        lat = f"{s.p95_total_latency_ms:.1f} ms"
        status = "✅ PASS" if s.thresholds_passed else "❌ FAIL"
        md.append(f"| `{name}` | {rec} | {mrr} | {cp} | {cr} | {lat} | {status} |")

    if regression_reports:
        md.extend([
            "",
            "## 2. Regression Detection Analysis",
            "",
            "| Configuration | Status | Evaluated Metrics | Regressions Flagged | Improvements |",
            "|:---|:---:|:---:|:---|:---|",
        ])
        for name, rep in regression_reports.items():
            status_badge = "✅ PASSED" if rep.passed else "⚠️ REGRESSION"
            degs = "; ".join(rep.degradations) if rep.degradations else "None"
            imps = "; ".join(rep.improvements) if rep.improvements else "None"
            md.append(f"| `{name}` | {status_badge} | {rep.total_metrics_evaluated} | {degs} | {imps} |")

    return "\n".join(md)


def main() -> int:
    args = parse_args()
    print("\n🔍 [DocuLens AI Evaluation Harness]")

    # 1. Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Gold dataset not found at '{dataset_path}'", file=sys.stderr)
        return 1

    dataset = load_gold_dataset(dataset_path)
    print(f" Loaded {dataset.total} gold queries from '{dataset_path.name}'")

    # 2. Setup stores & Index corpus
    stores = get_shared_stores()

    corpus_dir = Path(args.corpus_dir)
    if corpus_dir.exists() and corpus_dir.is_dir():
        pdf_files = sorted(corpus_dir.glob("*.pdf"))
        if pdf_files:
            print(f" Indexing {len(pdf_files)} evaluation corpus PDFs from '{corpus_dir}'...")
            corpus_stats = index_evaluation_corpus(stores, corpus_dir)
            for doc_id, s in corpus_stats.items():
                print(f"   [{doc_id}] dense={s['dense_chunks']}, bm25={s['bm25_chunks']}, visual={s.get('visual_pages', 0)}")
        else:
            # Fallback: single PDF via legacy --corpus-pdf
            corpus_pdf = Path(args.corpus_pdf)
            if corpus_pdf.exists():
                print(f" Indexing evaluation corpus '{corpus_pdf.name}' into shared stores...")
                doc_id = "doculens-architecture-v1"
                s = index_evaluation_document(stores, doc_id, corpus_pdf)
                print(f"   Indexed: {s['dense_chunks']} dense chunks, {s['bm25_chunks']} BM25 chunks, {s.get('visual_pages', 0)} visual pages")
            else:
                print("⚠️ Warning: No corpus PDFs found. Using pre-existing store data.")
    else:
        # Fallback: single PDF via legacy --corpus-pdf
        corpus_pdf = Path(args.corpus_pdf)
        if corpus_pdf.exists():
            print(f" Indexing evaluation corpus '{corpus_pdf.name}' into shared stores...")
            doc_id = "doculens-architecture-v1"
            s = index_evaluation_document(stores, doc_id, corpus_pdf)
            print(f"   Indexed: {s['dense_chunks']} dense chunks, {s['bm25_chunks']} BM25 chunks, {s.get('visual_pages', 0)} visual pages")
        else:
            print(f"⚠️ Warning: Corpus directory '{corpus_dir}' not found. Using pre-existing store data.")

    # 3. Parse K values
    k_vals = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    # 4. Execute evaluation
    results: dict[str, ExperimentResult] = {}
    configs_to_run = BASELINE_NAMES if args.config == "all" else [args.config]

    print(f"▶️ Executing evaluation for configurations: {', '.join(configs_to_run)}")
    for cfg_name in configs_to_run:
        res = run_baseline_evaluation(
            stores=stores,
            dataset=dataset,
            baseline_name=cfg_name,
            default_k=args.default_k,
            k_values=k_vals,
        )
        results[cfg_name] = res

    # 5. Terminal output
    print(format_summary_table(results))

    # 6. Regression check against baseline if requested
    regression_reports: dict[str, RegressionReport] = {}
    has_regression_failure = False

    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            print(f"🔬 Comparing against baseline '{baseline_path}'...")
            with baseline_path.open("r", encoding="utf-8") as fh:
                baseline_data = json.load(fh)

            for cfg_name, res in results.items():
                if cfg_name in baseline_data:
                    base_res = ExperimentResult.model_validate(baseline_data[cfg_name])
                    rep = compare_against_baseline(current=res, baseline=base_res)
                    regression_reports[cfg_name] = rep
                    if not rep.passed:
                        has_regression_failure = True
                        print(f"❌ [REGRESSION DETECTED] in '{cfg_name}':")
                        for deg in rep.degradations:
                            print(f"   - {deg}")
                    else:
                        print(f"✅ Baseline check passed for '{cfg_name}' (Status: {rep.status})")
        else:
            print(f"⚠️ Baseline file '{baseline_path}' not found. Skipping regression comparison.")

    # 7. Save baseline if requested
    if args.save_baseline:
        save_path = Path(args.save_baseline)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {k: v.model_dump(mode="json") for k, v in results.items()}
        with save_path.open("w", encoding="utf-8") as fh:
            json.dump(dump_data, fh, indent=2)
        print(f"💾 Saved baseline results to '{save_path}'")

    # 8. Save output results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {k: v.model_dump(mode="json") for k, v in results.items()}
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(dump_data, fh, indent=2)
        print(f"📄 Machine-readable results saved to '{out_path}'")

    # 9. Save markdown report if requested
    if args.markdown_output:
        md_path = Path(args.markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = format_markdown_report(results, regression_reports)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"📝 Markdown report saved to '{md_path}'")

    # 10. Exit code check
    if args.strict:
        any_gate_failed = any(not r.summary.thresholds_passed for r in results.values())
        if any_gate_failed or has_regression_failure:
            print("\n❌ Strict mode: Failure criteria breached.", file=sys.stderr)
            return 1

    print("✨ Evaluation complete.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
