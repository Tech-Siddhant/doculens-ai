from typing import Any, Sequence, TypeVar, Union
from app.core.config import settings
from app.schemas.retrieval import (
    EvidenceItem,
    EvidenceSelectionResult,
    RerankedCandidate,
    FusedCandidate,
    RetrievedChunk,
    RetrievedVisualPage,
)

AnyCandidate = Union[
    RerankedCandidate,
    FusedCandidate,
    RetrievedChunk,
    RetrievedVisualPage,
    dict[str, Any],
]

T = TypeVar("T")


class EvidenceSelector:
    """
    Service for selecting and deduplicating top-K evidence items from retrieval candidates.
    """

    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Helper to get attribute or dictionary key."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def _get_candidate_key(self, candidate: AnyCandidate) -> str:
        """
        Generate a unique key for deduplication.
        Prioritizes chunk_id if available, otherwise uses page_number.
        """
        doc_id = str(self._get_attr(candidate, "document_id", "unknown"))
        chunk_id = self._get_attr(candidate, "chunk_id")
        page_num = self._get_attr(candidate, "page_number", 0)

        if chunk_id:
            return f"chunk::{doc_id}::{chunk_id}"
        return f"page::{doc_id}::{page_num}"

    def select_evidence_items(
        self,
        candidates: Sequence[AnyCandidate],
        top_k: int | None = None,
        target_document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[EvidenceItem]:
        """
        Select the top-K evidence items from the candidates list.
        """
        # 1. Validate top_k
        if top_k is None:
            effective_k = settings.DEFAULT_EVIDENCE_TOP_K
        else:
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
                raise ValueError("top_k must be an integer >= 1.")
            effective_k = top_k

        if not candidates:
            return []

        seen_keys = set()
        evidence_items: list[EvidenceItem] = []

        # 2. Process candidates in order (preserving ranking)
        for i, candidate in enumerate(candidates):
            # Document isolation / filtering
            doc_id = str(self._get_attr(candidate, "document_id", ""))
            if target_document_id and doc_id != target_document_id:
                continue

            # Score threshold filtering
            score = float(self._get_attr(candidate, "score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue

            # Deduplication
            key = self._get_candidate_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Metadata extraction
            page_number = int(self._get_attr(candidate, "page_number", 1))
            chunk_id = self._get_attr(candidate, "chunk_id")
            chunk_index = self._get_attr(candidate, "chunk_index")
            text = self._get_attr(candidate, "text")
            image_url = self._get_attr(candidate, "image_url")
            
            # Source & Initial Rank/Score preservation
            sources = self._get_attr(candidate, "sources")
            if not sources:
                if isinstance(candidate, RetrievedVisualPage):
                    sources = ["visual"]
                elif isinstance(candidate, RetrievedChunk):
                    sources = ["dense"]
                else:
                    sources = ["unknown"]

            initial_rank = self._get_attr(candidate, "initial_rank")
            if initial_rank is None:
                initial_rank = self._get_attr(candidate, "rank", i + 1)

            initial_score = self._get_attr(candidate, "initial_score")
            if initial_score is None:
                initial_score = score

            # Create evidence item
            item = EvidenceItem(
                rank=len(evidence_items) + 1,
                score=score,
                initial_rank=initial_rank,
                initial_score=initial_score,
                document_id=doc_id,
                page_number=page_number,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                text=text,
                image_url=image_url,
                retrieval_type="evidence",
                sources=sources,
                raw_scores=self._get_attr(candidate, "raw_scores", {}),
                normalized_scores=self._get_attr(candidate, "normalized_scores", {}),
                metadata=self._get_attr(candidate, "metadata", {}),
            )
            evidence_items.append(item)

            # Stop once we reach K
            if len(evidence_items) >= effective_k:
                break

        return evidence_items

    def select_evidence(
        self,
        query: str,
        candidates: Sequence[AnyCandidate],
        top_k: int | None = None,
        target_document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> EvidenceSelectionResult:
        """
        Select evidence and return a structured result.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        evidence = self.select_evidence_items(
            candidates=candidates,
            top_k=top_k,
            target_document_id=target_document_id,
            score_threshold=score_threshold,
        )

        return EvidenceSelectionResult(
            query=query,
            evidence=evidence,
            top_k=top_k or settings.DEFAULT_EVIDENCE_TOP_K,
            total_candidates=len(candidates),
            total_selected=len(evidence),
        )


# Singleton instance
evidence_selector = EvidenceSelector()
