"""Evaluation metrics computation for DocuLens AI — Phase 6.3 & 6.4.

Provides deterministic, lightweight evaluation metrics across:
1. Retrieval Boundary: Recall@K, MRR@K, Hit@K, Context Precision, Context Recall.
2. Generation Boundary: Faithfulness, Answer Relevancy, Citation Precision, Citation Recall, Abstention Correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

from app.schemas.citation import Citation
from app.schemas.evaluation import (
    GenerationEvalMetrics,
    GoldQuery,
    RetrievalEvalMetrics,
)
from app.services.citation_validator import (
    CITATION_TAG_PATTERN,
    citation_validator,
    is_refusal_response,
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "then", "of", "to", "in",
    "on", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must", "it",
    "its", "they", "them", "their", "this", "that", "these", "those", "which", "what",
    "who", "whom", "whose", "as", "such", "than", "too", "very", "also", "just",
    "according", "mention", "mentioned", "described", "stated", "provides", "shows",
    "used", "using", "uses",
}


def _stem_token(token: str) -> str:
    """Normalize token by stripping common English inflectional suffixes."""
    t = token.lower()
    for suffix in ("ing", "tions", "tion", "ies", "ed", "es", "ly", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            t = t[:-len(suffix)]
            break
    if len(t) >= 4 and t.endswith("e"):
        t = t[:-1]
    return t


def _tokenize_significant(text: str) -> set[str]:
    """Extract lowercased alphanumeric words >= 2 chars, excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _token_stems(text: str) -> set[str]:
    """Extract stemmed set of significant tokens."""
    return {_stem_token(w) for w in _tokenize_significant(text)}


def _extract_claims(text: str) -> list[str]:
    """Split text into sentences/claims, stripping citation tags and blank segments."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[str] = []
    for s in raw_sentences:
        clean = CITATION_TAG_PATTERN.sub("", s).strip()
        if clean and len(clean) > 3:
            claims.append(s.strip())
    return claims


def extract_candidate_info(candidate: Any) -> tuple[str, int, str | None]:
    """Extract (document_id, page_number, chunk_id) from any candidate representation.

    Supports:
    - RetrievedChunk, RetrievedVisualPage, FusedCandidate, RerankedCandidate, EvidenceItem
    - Citation, dict representations, and integer page numbers
    """
    if isinstance(candidate, int):
        return ("", candidate, None)
    if isinstance(candidate, dict):
        doc_id = str(candidate.get("document_id", "") or "")
        page = int(candidate.get("page_number", 1) or 1)
        chunk_id = candidate.get("chunk_id")
        return (doc_id, page, chunk_id)

    doc_id = str(getattr(candidate, "document_id", "") or "")
    page = int(getattr(candidate, "page_number", 1) or 1)
    chunk_id = getattr(candidate, "chunk_id", None)
    return (doc_id, page, chunk_id)


def extract_evidence_text(evidence: Sequence[Any] | str) -> str:
    """Extract and concatenate text content from evidence items or raw string."""
    if isinstance(evidence, str):
        return evidence
    texts: list[str] = []
    for item in evidence:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            t = item.get("text") or item.get("ground_truth_evidence_text") or ""
            if t:
                texts.append(str(t))
        else:
            t = (
                getattr(item, "text", None)
                or getattr(item, "ground_truth_evidence_text", None)
                or ""
            )
            if t:
                texts.append(str(t))
    return " ".join(texts)


def is_candidate_relevant(
    candidate: Any,
    ground_truth_pages: Sequence[int],
    target_document_id: str | None = None,
    ground_truth_chunks: Sequence[str] | None = None,
) -> bool:
    """Check whether a single candidate matches ground-truth pages or chunks under document isolation."""
    doc_id, page, chunk_id = extract_candidate_info(candidate)

    if target_document_id and doc_id and doc_id != target_document_id:
        return False

    if ground_truth_chunks and chunk_id and chunk_id in ground_truth_chunks:
        return True

    return page in ground_truth_pages


def compute_recall_at_k(
    retrieved: Sequence[Any],
    ground_truth_pages: Sequence[int],
    k: int = 5,
    target_document_id: str | None = None,
    ground_truth_chunks: Sequence[str] | None = None,
) -> float:
    """Compute Recall@K: proportion of distinct ground-truth pages retrieved in top-K candidates."""
    gt_pages_set = set(ground_truth_pages)
    gt_chunks_set = set(ground_truth_chunks or [])

    if not gt_pages_set and not gt_chunks_set:
        return 0.0
    if k <= 0 or not retrieved:
        return 0.0

    top_k = retrieved[:k]
    matched_pages: set[int] = set()
    matched_chunks: set[str] = set()

    for item in top_k:
        doc_id, page, chunk_id = extract_candidate_info(item)
        if target_document_id and doc_id and doc_id != target_document_id:
            continue
        if page in gt_pages_set:
            matched_pages.add(page)
        if chunk_id and chunk_id in gt_chunks_set:
            matched_chunks.add(chunk_id)

    if gt_pages_set:
        return round(len(matched_pages) / len(gt_pages_set), 6)
    elif gt_chunks_set:
        return round(len(matched_chunks) / len(gt_chunks_set), 6)
    return 0.0


def compute_mrr_at_k(
    retrieved: Sequence[Any],
    ground_truth_pages: Sequence[int],
    k: int = 5,
    target_document_id: str | None = None,
    ground_truth_chunks: Sequence[str] | None = None,
) -> float:
    """Compute MRR@K: reciprocal rank of the first relevant candidate in top-K."""
    if not ground_truth_pages and not ground_truth_chunks:
        return 0.0
    if k <= 0 or not retrieved:
        return 0.0

    top_k = retrieved[:k]
    for rank_idx, item in enumerate(top_k, start=1):
        if is_candidate_relevant(
            item,
            ground_truth_pages,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        ):
            return round(1.0 / rank_idx, 6)

    return 0.0


def compute_hit_at_k(
    retrieved: Sequence[Any],
    ground_truth_pages: Sequence[int],
    k: int = 5,
    target_document_id: str | None = None,
    ground_truth_chunks: Sequence[str] | None = None,
) -> bool:
    """Compute Hit@K: whether at least one ground-truth page was retrieved in top-K candidates."""
    return (
        compute_mrr_at_k(
            retrieved,
            ground_truth_pages,
            k=k,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        )
        > 0.0
    )


def compute_context_precision(
    retrieved: Sequence[Any],
    ground_truth_pages: Sequence[int],
    k: int = 5,
    target_document_id: str | None = None,
    ground_truth_chunks: Sequence[str] | None = None,
) -> float:
    """Compute Context Precision@K: precision of retrieved evidence ranking.

    Formula:
        Context Precision@K = sum_{i=1}^k (Precision@i * I(candidate_i is relevant)) / (Total relevant in top-k)
    """
    if not ground_truth_pages and not ground_truth_chunks:
        return 0.0
    if k <= 0 or not retrieved:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = 0
    precision_sum = 0.0

    for idx, item in enumerate(top_k, start=1):
        if is_candidate_relevant(
            item,
            ground_truth_pages,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        ):
            relevant_count += 1
            precision_at_i = relevant_count / idx
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0
    return round(precision_sum / relevant_count, 6)


def compute_context_recall(
    retrieved: Sequence[Any] | str,
    key_reference_facts: Sequence[str] | None = None,
    ground_truth_evidence_text: str = "",
    ground_truth_pages: Sequence[int] | None = None,
    target_document_id: str | None = None,
) -> float:
    """Compute Context Recall: proportion of reference facts covered in retrieved context."""
    facts = [f.strip() for f in (key_reference_facts or []) if f.strip()]
    if facts:
        context_text = extract_evidence_text(retrieved).lower()
        if not context_text.strip():
            return 0.0
        covered = 0
        context_stems = _token_stems(context_text)
        for fact in facts:
            fact_lower = fact.lower()
            if fact_lower in context_text:
                covered += 1
                continue
            fact_stems = _token_stems(fact_lower)
            if not fact_stems:
                covered += 1
                continue
            match_ratio = len(fact_stems & context_stems) / len(fact_stems)
            if match_ratio >= 0.50:
                covered += 1
        return round(covered / len(facts), 6)

    if ground_truth_evidence_text.strip():
        context_text = extract_evidence_text(retrieved).lower()
        if not context_text.strip():
            return 0.0
        gt_stems = _token_stems(ground_truth_evidence_text.lower())
        if not gt_stems:
            return 1.0
        context_stems = _token_stems(context_text)
        return round(min(1.0, len(gt_stems & context_stems) / len(gt_stems)), 6)

    if ground_truth_pages and isinstance(retrieved, (list, tuple)):
        return compute_recall_at_k(
            retrieved=retrieved,
            ground_truth_pages=ground_truth_pages,
            k=len(retrieved),
            target_document_id=target_document_id,
        )

    return 0.0


def compute_faithfulness(
    answer: str,
    evidence: Sequence[Any] | str,
    question: str = "",
    is_refusal: bool = False,
) -> float:
    """Compute Faithfulness: proportion of claims in answer entailed by retrieved context."""
    if is_refusal or is_refusal_response(answer):
        return 1.0

    clean_answer = answer.strip()
    if not clean_answer:
        return 0.0

    context_text = extract_evidence_text(evidence).lower()
    if not context_text.strip():
        return 0.0

    claims = _extract_claims(clean_answer)
    if not claims:
        return 1.0

    entailed_count = 0
    context_stems = _token_stems(context_text)
    q_stems = _token_stems(question.lower()) if question else set()

    for claim in claims:
        claim_clean = CITATION_TAG_PATTERN.sub("", claim).strip().lower()
        if not claim_clean:
            entailed_count += 1
            continue
        if claim_clean in context_text:
            entailed_count += 1
            continue
        stems = _token_stems(claim_clean)
        if not stems:
            entailed_count += 1
            continue

        novel_stems = stems - q_stems
        if novel_stems:
            novel_overlap = len(novel_stems & context_stems) / len(novel_stems)
            if novel_overlap >= 0.50:
                entailed_count += 1
                continue

        overlap = len(stems & context_stems) / len(stems)
        if overlap >= 0.50:
            entailed_count += 1

    return round(entailed_count / len(claims), 6)


def compute_answer_relevancy(
    question: str,
    answer: str,
    ground_truth_answer: str = "",
    is_answerable: bool = True,
    is_refusal: bool = False,
) -> float:
    """Compute Answer Relevancy: how well the answer addresses the question."""
    refusal = is_refusal or is_refusal_response(answer)
    if not is_answerable:
        return 1.0 if refusal else 0.0
    if is_answerable and refusal:
        return 0.0

    clean_ans = answer.strip().lower()
    if not clean_ans:
        return 0.0

    q_stems = _token_stems(question.lower())
    ans_stems = _token_stems(clean_ans)
    if not ans_stems:
        return 0.0

    q_cov = len(q_stems & ans_stems) / len(q_stems) if q_stems else 1.0

    if ground_truth_answer.strip():
        gt_stems = _token_stems(ground_truth_answer.lower())
        if gt_stems:
            gt_cov = len(gt_stems & ans_stems) / len(gt_stems)
            ans_prec = len(gt_stems & ans_stems) / len(ans_stems)
            gt_f1 = (
                (2 * gt_cov * ans_prec / (gt_cov + ans_prec))
                if (gt_cov + ans_prec) > 0
                else 0.0
            )
            score = max(q_cov, 0.5 * q_cov + 0.5 * gt_f1, gt_cov)
            return round(min(1.0, score), 6)

    return round(min(1.0, q_cov), 6)


def compute_citation_precision(
    citations: Sequence[Any],
    valid_evidence_pages: Sequence[int] | None = None,
    target_document_id: str | None = None,
    is_refusal: bool = False,
) -> float:
    """Compute Citation Precision: valid citations / total citations."""
    if is_refusal:
        return 1.0
    if not citations:
        return 0.0

    valid_count = 0
    for cit in citations:
        doc_id, page, _ = extract_candidate_info(cit)
        if target_document_id and doc_id and doc_id != target_document_id:
            continue
        if page < 1:
            continue
        if valid_evidence_pages is not None and page not in valid_evidence_pages:
            continue
        valid_count += 1

    return round(valid_count / len(citations), 6)


def compute_citation_recall(
    citations: Sequence[Any],
    ground_truth_pages: Sequence[int],
    target_document_id: str | None = None,
) -> float:
    """Compute Citation Recall: |GroundTruthPages ∩ CitedPages| / |GroundTruthPages|."""
    if not ground_truth_pages:
        return 1.0
    if not citations:
        return 0.0

    cited_pages: set[int] = set()
    for cit in citations:
        doc_id, page, _ = extract_candidate_info(cit)
        if target_document_id and doc_id and doc_id != target_document_id:
            continue
        cited_pages.add(page)

    gt_pages_set = set(ground_truth_pages)
    return round(len(gt_pages_set & cited_pages) / len(gt_pages_set), 6)


def compute_abstention_correctness(answer: str, is_answerable: bool) -> bool:
    """Check if model correctly abstained when unanswerable, or answered when answerable."""
    refusal = is_refusal_response(answer)
    if not is_answerable:
        return refusal
    return not refusal


def compute_retrieval_metrics(
    retrieved: Sequence[Any],
    ground_truth_pages: Sequence[int],
    k_values: Sequence[int] = (1, 3, 5, 10),
    default_k: int = 5,
    target_document_id: str | None = None,
    ground_truth_chunks: Sequence[str] | None = None,
    key_reference_facts: Sequence[str] | None = None,
    ground_truth_evidence_text: str = "",
) -> RetrievalEvalMetrics:
    """Compute full RetrievalEvalMetrics for a query across multi-K values."""
    retrieved_pages: list[int] = []
    retrieved_chunk_ids: list[str] = []

    for item in retrieved:
        doc_id, page, chunk_id = extract_candidate_info(item)
        if target_document_id and doc_id and doc_id != target_document_id:
            continue
        retrieved_pages.append(page)
        if chunk_id:
            retrieved_chunk_ids.append(chunk_id)

    k_metrics: dict[int, dict[str, float]] = {}
    for k in k_values:
        rec = compute_recall_at_k(
            retrieved=retrieved,
            ground_truth_pages=ground_truth_pages,
            k=k,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        )
        mrr = compute_mrr_at_k(
            retrieved=retrieved,
            ground_truth_pages=ground_truth_pages,
            k=k,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        )
        hit = 1.0 if rec > 0.0 else 0.0
        k_metrics[k] = {"recall": rec, "mrr": mrr, "hit": hit}

    if default_k in k_metrics:
        default_rec = k_metrics[default_k]["recall"]
        default_mrr = k_metrics[default_k]["mrr"]
    else:
        default_rec = compute_recall_at_k(
            retrieved,
            ground_truth_pages,
            k=default_k,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        )
        default_mrr = compute_mrr_at_k(
            retrieved,
            ground_truth_pages,
            k=default_k,
            target_document_id=target_document_id,
            ground_truth_chunks=ground_truth_chunks,
        )

    context_precision = compute_context_precision(
        retrieved=retrieved,
        ground_truth_pages=ground_truth_pages,
        k=default_k,
        target_document_id=target_document_id,
        ground_truth_chunks=ground_truth_chunks,
    )

    context_recall = compute_context_recall(
        retrieved=retrieved[:default_k],
        key_reference_facts=key_reference_facts,
        ground_truth_evidence_text=ground_truth_evidence_text,
        ground_truth_pages=ground_truth_pages,
        target_document_id=target_document_id,
    )

    return RetrievalEvalMetrics(
        recall_at_k=default_rec,
        mrr_at_k=default_mrr,
        context_precision=context_precision,
        context_recall=context_recall,
        hit_at_k=(default_rec > 0.0),
        retrieved_pages=retrieved_pages[:default_k],
        retrieved_chunk_ids=retrieved_chunk_ids[:default_k],
        k_metrics=k_metrics,
    )


def compute_generation_metrics(
    query: GoldQuery,
    answer: str,
    evidence: Sequence[Any] | str,
    citations: Sequence[Citation] | None = None,
) -> GenerationEvalMetrics:
    """Compute generation evaluation metrics deterministically."""
    from app.evaluation.judges import DeterministicGenerationJudge

    judge = DeterministicGenerationJudge()
    evidence_seq = [evidence] if isinstance(evidence, str) else list(evidence)
    return judge.evaluate(
        query=query,
        answer=answer,
        evidence=evidence_seq,
        citations=citations,
    )
