"""Cross-encoder reranking service for DocuLens AI — Phase 5.1 baseline.

Reranks hybrid retrieval candidates by scoring (query, candidate-text) pairs
with a lightweight ONNX cross-encoder. No new dependencies: uses
fastembed.rerank.cross_encoder which ships with the already-installed
fastembed package and runs on CPU via ONNX Runtime.

Chosen model: Xenova/ms-marco-MiniLM-L-6-v2
  - 80 MB ONNX, Apache 2.0, CPU-only, no CUDA required.
  - Smallest available option.

# ponytail: single global model singleton; replace with per-request lazy init or pool if multi-threaded throughput matters.
"""

from collections.abc import Sequence
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.core.config import settings
from app.schemas.retrieval import (
    FusedCandidate,
    RerankedCandidate,
    RerankResult,
    RetrievedChunk,
    RetrievedVisualPage,
)

# Type alias: anything the hybrid pipeline may return
AnyCandidate = FusedCandidate | RetrievedChunk | RetrievedVisualPage | dict[str, Any]


def _candidate_text(candidate: AnyCandidate) -> str:
    """Extract text for reranking. Visual-only pages fall back to a
    deterministic string so the cross-encoder still sees them.
    # ponytail: caption/OCR fallback — attach OCR text in metadata when available.
    """
    text: str | None = None
    if isinstance(candidate, dict):
        text = candidate.get("text") or None
        if not text:
            doc_id = candidate.get("document_id", "")
            page = candidate.get("page_number", 1)
            text = f"Document {doc_id} Page {page}"
    else:
        text = getattr(candidate, "text", None) or None
        if not text:
            doc_id = getattr(candidate, "document_id", "")
            page = getattr(candidate, "page_number", 1)
            # Check metadata for a caption or description
            meta: dict[str, Any] = getattr(candidate, "metadata", {}) or {}
            text = meta.get("caption") or meta.get("description") or f"Document {doc_id} Page {page}"
    return text


def _get_attr(candidate: AnyCandidate, attr: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(attr, default)
    return getattr(candidate, attr, default)


class CrossEncoderReranker:
    """Rerank retrieval candidates with a cross-encoder model.

    Input:  query string + sequence of retrieval candidates (FusedCandidate /
            RetrievedChunk / RetrievedVisualPage or plain dicts).
    Output: list[RerankedCandidate] sorted by cross-encoder relevance score,
            preserving all original retrieval metadata and the original rank.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.batch_size = batch_size or settings.RERANKER_BATCH_SIZE
        self._model: TextCrossEncoder | None = None

    @property
    def model(self) -> TextCrossEncoder:
        """Lazy-load the cross-encoder model on first access."""
        if self._model is None:
            try:
                self._model = TextCrossEncoder(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load reranker model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score (query, text) pairs with the cross-encoder.

        Returns scores in the same order as texts.
        The cross-encoder produces raw logits; higher = more relevant.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Reranker query must be a non-empty string.")
        if not texts:
            return []
        try:
            scores = list(self.model.rerank(query, list(texts), batch_size=self.batch_size))
        except Exception as exc:
            raise RuntimeError(f"Reranker inference failed: {exc}") from exc
        return [float(s) for s in scores]

    def rerank_candidates(
        self,
        query: str,
        candidates: Sequence[AnyCandidate],
        top_k: int | None = None,
        target_document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RerankedCandidate]:
        """Rerank candidates for query and return a sorted list.

        Args:
            query: search query string.
            candidates: retrieval candidates from any upstream stage.
            top_k: maximum number of results to return. Defaults to
                   settings.DEFAULT_RERANK_TOP_K.
            target_document_id: when set, enforces strict isolation.
            score_threshold: minimum cross-encoder score to include.

        Returns:
            list[RerankedCandidate] sorted descending by reranker score;
            original rank available in RerankedCandidate.initial_rank.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Reranker query must be a non-empty string.")
        if not candidates:
            return []

        effective_top_k = top_k if top_k is not None else settings.DEFAULT_RERANK_TOP_K

        # Apply document isolation filter before scoring
        filtered: list[AnyCandidate] = [
            c for c in candidates
            if target_document_id is None
            or str(_get_attr(c, "document_id", "")) == target_document_id
        ]
        if not filtered:
            return []

        texts = [_candidate_text(c) for c in filtered]
        scores = self.score_pairs(query, texts)

        reranked: list[RerankedCandidate] = []
        for idx, (candidate, rerank_score) in enumerate(zip(filtered, scores, strict=True)):
            if score_threshold is not None and rerank_score < score_threshold:
                continue

            # Derive initial rank: use the candidate's own rank field if present,
            # otherwise fall back to the 1-based position in the input list.
            initial_rank: int = int(_get_attr(candidate, "rank", idx + 1))
            initial_score: float = float(_get_attr(candidate, "score", 0.0))

            sources: list[str] = list(_get_attr(candidate, "sources", []) or [])
            if isinstance(candidate, RetrievedChunk) and not sources:
                sources = [_get_attr(candidate, "retrieval_type", "dense")]
            elif isinstance(candidate, RetrievedVisualPage) and not sources:
                sources = ["visual"]

            reranked.append(
                RerankedCandidate(
                    # placeholder rank; re-assigned after sorting
                    rank=idx + 1,
                    score=rerank_score,
                    initial_rank=initial_rank,
                    initial_score=initial_score,
                    document_id=str(_get_attr(candidate, "document_id", "")),
                    page_number=int(_get_attr(candidate, "page_number", 1)),
                    chunk_id=_get_attr(candidate, "chunk_id"),
                    chunk_index=_get_attr(candidate, "chunk_index"),
                    text=_get_attr(candidate, "text"),
                    image_url=_get_attr(candidate, "image_url"),
                    retrieval_type="reranked",
                    sources=sources,
                    raw_scores=dict(_get_attr(candidate, "raw_scores", {}) or {}),
                    normalized_scores=dict(_get_attr(candidate, "normalized_scores", {}) or {}),
                    metadata=dict(_get_attr(candidate, "metadata", {}) or {}),
                )
            )

        # Sort: descending score, then ascending initial_rank for a stable tie-break
        reranked.sort(key=lambda c: (-c.score, c.initial_rank))

        # Cap and re-index ranks
        reranked = reranked[:effective_top_k]
        for new_rank, item in enumerate(reranked, start=1):
            item.rank = new_rank

        return reranked

    def rerank(
        self,
        query: str,
        candidates: Sequence[AnyCandidate],
        top_k: int | None = None,
        target_document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> RerankResult:
        """Rerank candidates and return a structured RerankResult."""
        effective_top_k = top_k if top_k is not None else settings.DEFAULT_RERANK_TOP_K
        results = self.rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=effective_top_k,
            target_document_id=target_document_id,
            score_threshold=score_threshold,
        )
        return RerankResult(
            query=query,
            document_id=target_document_id,
            model_name=self.model_name,
            top_k=effective_top_k,
            total_results=len(results),
            results=results,
        )


reranker = CrossEncoderReranker()
