"""Hybrid multi-modal retrieval fusion service for DocuLens AI."""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from app.core.config import settings
from app.schemas.retrieval import (
    FusedCandidate,
    HybridRetrievalResult,
    ModalityWeights,
    RetrievedChunk,
    RetrievedVisualPage,
)
from app.services.bm25 import BM25Retriever, bm25_retriever
from app.services.normalizer import normalize_retrieval_results
from app.services.retriever import DenseRetriever, retriever
from app.services.visual_embedder import VisualEmbeddingService, visual_embedding_service
from app.services.visual_vector_store import VisualQdrantVectorStore, visual_vector_store

logger = logging.getLogger(__name__)


def get_candidate_key(item: RetrievedChunk | RetrievedVisualPage | dict[str, Any]) -> str:
    """Generate deterministic unique identity key for a retrieval candidate."""
    if isinstance(item, RetrievedChunk):
        return f"chunk::{item.document_id}::{item.chunk_id}"
    elif isinstance(item, RetrievedVisualPage):
        return f"page::{item.document_id}::{item.page_number}"
    elif isinstance(item, dict):
        doc_id = item.get("document_id", "")
        if "chunk_id" in item and item["chunk_id"]:
            return f"chunk::{doc_id}::{item['chunk_id']}"
        return f"page::{doc_id}::{item.get('page_number', 1)}"
    else:
        doc_id = getattr(item, "document_id", "")
        chunk_id = getattr(item, "chunk_id", None)
        if chunk_id:
            return f"chunk::{doc_id}::{chunk_id}"
        return f"page::{doc_id}::{getattr(item, 'page_number', 1)}"


def collect_and_merge_candidates(
    dense_results: Sequence[RetrievedChunk],
    bm25_results: Sequence[RetrievedChunk],
    visual_results: Sequence[RetrievedVisualPage],
    target_document_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect candidates across retrieval channels and merge duplicates."""
    candidates: dict[str, dict[str, Any]] = {}

    channels: list[tuple[str, Sequence[Any]]] = [
        ("dense", dense_results),
        ("bm25", bm25_results),
        ("visual", visual_results),
    ]

    for channel_name, results in channels:
        for item in results:
            doc_id = getattr(item, "document_id", "")
            if target_document_id and doc_id != target_document_id:
                # Enforce strict document isolation
                continue

            key = get_candidate_key(item)
            if key not in candidates:
                candidates[key] = {
                    "candidate_key": key,
                    "document_id": doc_id,
                    "page_number": getattr(item, "page_number", 1),
                    "chunk_id": getattr(item, "chunk_id", None),
                    "chunk_index": getattr(item, "chunk_index", None),
                    "text": getattr(item, "text", None),
                    "image_url": getattr(item, "image_url", None),
                    "sources": [],
                    "raw_scores": {},
                    "normalized_scores": {},
                    "ranks": {},
                    "metadata": {},
                }

            record = candidates[key]
            if channel_name not in record["sources"]:
                record["sources"].append(channel_name)

            raw_s = getattr(item, "raw_score", getattr(item, "score", 0.0))
            norm_s = getattr(item, "normalized_score", 0.0)
            rank_s = getattr(item, "rank", 1)

            record["raw_scores"][channel_name] = float(raw_s) if raw_s is not None else 0.0
            record["normalized_scores"][channel_name] = float(norm_s) if norm_s is not None else 0.0
            record["ranks"][channel_name] = int(rank_s)

            # Merge text/image/metadata if not already present
            if not record["text"] and getattr(item, "text", None):
                record["text"] = getattr(item, "text", None)
            if not record["image_url"] and getattr(item, "image_url", None):
                record["image_url"] = getattr(item, "image_url", None)
            if hasattr(item, "metadata") and item.metadata:
                record["metadata"].update(item.metadata)

    return candidates


def fuse_weighted(
    candidates: dict[str, dict[str, Any]],
    weights: ModalityWeights | dict[str, float] | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[FusedCandidate]:
    """Combine candidates using weighted linear fusion of normalized scores.

    Formula:
        fused_score = w_dense * score_norm_dense + w_bm25 * score_norm_bm25 + w_visual * score_norm_visual
    """
    if isinstance(weights, ModalityWeights):
        w_dict = weights.normalized_dict()
    elif isinstance(weights, dict):
        w_dense = float(weights.get("dense", settings.DEFAULT_WEIGHT_DENSE))
        w_bm25 = float(weights.get("bm25", settings.DEFAULT_WEIGHT_BM25))
        w_visual = float(weights.get("visual", settings.DEFAULT_WEIGHT_VISUAL))
        if w_dense < 0.0 or w_bm25 < 0.0 or w_visual < 0.0:
            raise ValueError("Modality weights must be non-negative.")
        total_w = w_dense + w_bm25 + w_visual
        if total_w <= 0.0:
            raise ValueError("At least one modality weight must be strictly positive.")
        w_dict = {
            "dense": round(w_dense / total_w, 6),
            "bm25": round(w_bm25 / total_w, 6),
            "visual": round(w_visual / total_w, 6),
        }
    else:
        w_obj = ModalityWeights(
            dense=settings.DEFAULT_WEIGHT_DENSE,
            bm25=settings.DEFAULT_WEIGHT_BM25,
            visual=settings.DEFAULT_WEIGHT_VISUAL,
        )
        w_dict = w_obj.normalized_dict()

    scored_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for key, cand in candidates.items():
        score = 0.0
        for mod, weight in w_dict.items():
            score += weight * cand["normalized_scores"].get(mod, 0.0)
        score = round(score, 6)

        if score_threshold is not None and score < score_threshold:
            continue

        scored_candidates.append((score, key, cand))

    # Sort descending by score, tie-break by candidate key
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))

    fused_results: list[FusedCandidate] = []
    for rank, (score, _, cand) in enumerate(scored_candidates[:top_k], start=1):
        fused_results.append(
            FusedCandidate(
                rank=rank,
                score=score,
                document_id=cand["document_id"],
                page_number=cand["page_number"],
                chunk_id=cand["chunk_id"],
                chunk_index=cand["chunk_index"],
                text=cand["text"],
                image_url=cand["image_url"],
                retrieval_type="hybrid",
                sources=cand["sources"],
                raw_scores=cand["raw_scores"],
                normalized_scores=cand["normalized_scores"],
                metadata=cand["metadata"],
            )
        )

    return fused_results


def fuse_rrf(
    candidates: dict[str, dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[FusedCandidate]:
    """Combine candidates using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{m in sources(d)} 1.0 / (k + rank_m(d))
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

    scored_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for key, cand in candidates.items():
        rrf_score = 0.0
        for mod in cand["sources"]:
            rank_m = cand["ranks"].get(mod, 1)
            rrf_score += 1.0 / (rrf_k + rank_m)
        rrf_score = round(rrf_score, 6)

        if score_threshold is not None and rrf_score < score_threshold:
            continue

        scored_candidates.append((rrf_score, key, cand))

    # Sort descending by RRF score, tie-break by candidate key
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))

    fused_results: list[FusedCandidate] = []
    for rank, (score, _, cand) in enumerate(scored_candidates[:top_k], start=1):
        fused_results.append(
            FusedCandidate(
                rank=rank,
                score=score,
                document_id=cand["document_id"],
                page_number=cand["page_number"],
                chunk_id=cand["chunk_id"],
                chunk_index=cand["chunk_index"],
                text=cand["text"],
                image_url=cand["image_url"],
                retrieval_type="hybrid",
                sources=cand["sources"],
                raw_scores=cand["raw_scores"],
                normalized_scores=cand["normalized_scores"],
                metadata=cand["metadata"],
            )
        )

    return fused_results


class HybridRetriever:
    """Service combining Dense, BM25, and Visual retrieval into unified ranked candidates."""

    def __init__(
        self,
        dense_retriever_inst: DenseRetriever | None = None,
        bm25_retriever_inst: BM25Retriever | None = None,
        visual_embedder_inst: VisualEmbeddingService | None = None,
        visual_store_inst: VisualQdrantVectorStore | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever_inst or retriever
        self.bm25_retriever = bm25_retriever_inst or bm25_retriever
        self.visual_embedder = visual_embedder_inst or visual_embedding_service
        self.visual_store = visual_store_inst or visual_vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        strategy: Literal["weighted", "rrf"] | str = "weighted",
        weights: ModalityWeights | dict[str, float] | None = None,
        rrf_k: int = 60,
        score_threshold: float | None = None,
        include_dense: bool = True,
        include_bm25: bool = True,
        include_visual: bool = True,
    ) -> HybridRetrievalResult:
        """Execute multi-channel hybrid retrieval and fusion."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        target_top_k = top_k if top_k is not None else settings.DEFAULT_RETRIEVAL_TOP_K
        if target_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if strategy not in ("weighted", "rrf"):
            raise ValueError(
                f"Invalid fusion strategy: '{strategy}'. Supported strategies are 'weighted' and 'rrf'."
            )

        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1.")

        # 1. Execute Dense Retrieval (if enabled)
        dense_raw: list[RetrievedChunk] = []
        if include_dense:
            try:
                dense_raw = self.dense_retriever.retrieve(
                    query=query,
                    top_k=target_top_k,
                    document_id=document_id,
                ).results
            except Exception as exc:
                logger.warning(
                    f"Dense retrieval failed: {exc}. Falling back to remaining channels.",
                    extra={"query": query, "document_id": document_id, "error": str(exc)},
                )
                dense_raw = []

        # 2. Execute BM25 Retrieval (if enabled)
        bm25_raw: list[RetrievedChunk] = []
        if include_bm25:
            try:
                bm25_raw = self.bm25_retriever.retrieve(
                    query=query,
                    top_k=target_top_k,
                    document_id=document_id,
                ).results
            except Exception as exc:
                logger.warning(
                    f"BM25 retrieval failed: {exc}. Falling back to remaining channels.",
                    extra={"query": query, "document_id": document_id, "error": str(exc)},
                )
                bm25_raw = []

        # 3. Execute Visual Retrieval (if enabled and available)
        visual_raw: list[RetrievedVisualPage] = []
        if include_visual:
            try:
                query_vector = self.visual_embedder.embed_visual_query(query)
                if query_vector:
                    visual_raw = self.visual_store.search_visual(
                        query_vector=query_vector,
                        top_k=target_top_k,
                        document_id=document_id,
                    )
            except Exception as exc:
                logger.warning(
                    f"Visual retrieval failed: {exc}. Falling back to remaining channels.",
                    extra={"query": query, "document_id": document_id, "error": str(exc)},
                )
                visual_raw = []

        # 4. Normalize Scores per Modality
        dense_norm = normalize_retrieval_results(dense_raw)
        bm25_norm = normalize_retrieval_results(bm25_raw)
        visual_norm = normalize_retrieval_results(visual_raw)

        # 5. Merge Candidate Set
        candidates = collect_and_merge_candidates(
            dense_results=dense_norm,
            bm25_results=bm25_norm,
            visual_results=visual_norm,
            target_document_id=document_id,
        )

        # 6. Apply Selected Fusion Strategy
        used_weights: dict[str, float] | None = None
        if strategy == "weighted":
            fused = fuse_weighted(
                candidates=candidates,
                weights=weights,
                top_k=target_top_k,
                score_threshold=score_threshold,
            )
            if isinstance(weights, ModalityWeights):
                used_weights = weights.normalized_dict()
            elif isinstance(weights, dict):
                w_obj = ModalityWeights(**weights)
                used_weights = w_obj.normalized_dict()
            else:
                used_weights = ModalityWeights(
                    dense=settings.DEFAULT_WEIGHT_DENSE,
                    bm25=settings.DEFAULT_WEIGHT_BM25,
                    visual=settings.DEFAULT_WEIGHT_VISUAL,
                ).normalized_dict()
        else:
            fused = fuse_rrf(
                candidates=candidates,
                rrf_k=rrf_k,
                top_k=target_top_k,
                score_threshold=score_threshold,
            )

        return HybridRetrievalResult(
            query=query,
            document_id=document_id,
            top_k=target_top_k,
            strategy=strategy,
            total_results=len(fused),
            results=fused,
            weights=used_weights if strategy == "weighted" else None,
            rrf_k=rrf_k if strategy == "rrf" else None,
        )

    def compare_strategies(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        weights: ModalityWeights | dict[str, float] | None = None,
        rrf_k: int = 60,
    ) -> dict[str, HybridRetrievalResult]:
        """Execute deterministic comparison across retrieval modality subsets and fusion strategies."""
        return {
            "dense_only": self.retrieve(
                query=query,
                top_k=top_k,
                document_id=document_id,
                strategy="weighted",
                weights={"dense": 1.0, "bm25": 0.0, "visual": 0.0},
                include_dense=True,
                include_bm25=False,
                include_visual=False,
            ),
            "dense_bm25_weighted": self.retrieve(
                query=query,
                top_k=top_k,
                document_id=document_id,
                strategy="weighted",
                weights={"dense": 0.7, "bm25": 0.3, "visual": 0.0},
                include_dense=True,
                include_bm25=True,
                include_visual=False,
            ),
            "dense_bm25_rrf": self.retrieve(
                query=query,
                top_k=top_k,
                document_id=document_id,
                strategy="rrf",
                rrf_k=rrf_k,
                include_dense=True,
                include_bm25=True,
                include_visual=False,
            ),
            "dense_bm25_visual_weighted": self.retrieve(
                query=query,
                top_k=top_k,
                document_id=document_id,
                strategy="weighted",
                weights=weights,
                include_dense=True,
                include_bm25=True,
                include_visual=True,
            ),
            "dense_bm25_visual_rrf": self.retrieve(
                query=query,
                top_k=top_k,
                document_id=document_id,
                strategy="rrf",
                rrf_k=rrf_k,
                include_dense=True,
                include_bm25=True,
                include_visual=True,
            ),
        }


hybrid_retriever = HybridRetriever()
