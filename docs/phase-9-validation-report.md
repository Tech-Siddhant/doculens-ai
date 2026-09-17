# DocuLens AI — Automated Evaluation Report

## 1. Summary of Baseline Configurations

| Configuration | Recall@5 | MRR@5 | Context Precision | Context Recall | Latency (p95) | Gate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `dense` | 1.0000 | 0.9556 | 0.8327 | 1.0000 | 0.0 ms | ✅ PASS |
| `dense_bm25` | 1.0000 | 0.9889 | 0.8530 | 1.0000 | 0.0 ms | ✅ PASS |
| `dense_bm25_visual` | 1.0000 | 1.0000 | 0.8447 | 1.0000 | 0.0 ms | ✅ PASS |
| `hybrid_reranked` | 1.0000 | 1.0000 | 0.8399 | 1.0000 | 0.0 ms | ✅ PASS |