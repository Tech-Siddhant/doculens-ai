"""BM25 sparse lexical retrieval service for DocuLens AI."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from app.core.config import settings
from app.schemas.document import DocumentChunk
from app.schemas.retrieval import BM25IndexStats, RetrievalResult, RetrievedChunk

# Regex preserving alphanumeric words, numbers, decimals, and internal hyphens/underscores/dots
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Deterministic tokenization preserving technical identifiers, acronyms, and numbers.

    Rules:
    - Lowercases input string.
    - Strips surrounding punctuation, brackets, quotes, and whitespace.
    - Preserves internal hyphens, underscores, dots, and slashes (e.g. ISO-27001, ABC-123, GPT-4, 42.7, 2024-Q4).
    - Stop words are retained to prevent destruction of domain codes and exact technical phrases.
    """
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


class BM25OkapiIndex:
    """In-memory BM25 Okapi index supporting document-level isolation."""

    def __init__(
        self,
        k1: float | None = None,
        b: float | None = None,
        epsilon: float | None = None,
    ) -> None:
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        self.epsilon = epsilon if epsilon is not None else settings.BM25_EPSILON

        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_ids: dict[str, set[str]] = defaultdict(set)
        self._chunk_tokens: dict[str, list[str]] = {}
        self._chunk_term_counts: dict[str, Counter[str]] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._inverted_index: dict[str, set[str]] = defaultdict(set)

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Index or re-index a sequence of DocumentChunk objects."""
        if not chunks:
            return 0

        count = 0
        for chunk in chunks:
            # If chunk already exists, remove previous references first
            if chunk.chunk_id in self._chunks:
                self._remove_chunk(chunk.chunk_id)

            tokens = tokenize(chunk.text)
            term_counts = Counter(tokens)

            self._chunks[chunk.chunk_id] = chunk
            self._doc_ids[chunk.document_id].add(chunk.chunk_id)
            self._chunk_tokens[chunk.chunk_id] = tokens
            self._chunk_term_counts[chunk.chunk_id] = term_counts
            self._chunk_lengths[chunk.chunk_id] = len(tokens)

            for term in term_counts:
                self._inverted_index[term].add(chunk.chunk_id)

            count += 1

        return count

    def _remove_chunk(self, chunk_id: str) -> None:
        """Helper to remove a single chunk from the index."""
        chunk = self._chunks.pop(chunk_id, None)
        if not chunk:
            return

        if chunk.document_id in self._doc_ids:
            self._doc_ids[chunk.document_id].discard(chunk_id)
            if not self._doc_ids[chunk.document_id]:
                del self._doc_ids[chunk.document_id]

        tokens = self._chunk_tokens.pop(chunk_id, [])
        self._chunk_term_counts.pop(chunk_id, None)
        self._chunk_lengths.pop(chunk_id, None)

        for term in set(tokens):
            if term in self._inverted_index:
                self._inverted_index[term].discard(chunk_id)
                if not self._inverted_index[term]:
                    del self._inverted_index[term]

    def delete_by_document(self, document_id: str) -> int:
        """Delete all indexed chunks associated with a specific document_id."""
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        chunk_ids = list(self._doc_ids.get(document_id, set()))
        for chunk_id in chunk_ids:
            self._remove_chunk(chunk_id)

        return len(chunk_ids)

    def count_chunks(self, document_id: str | None = None) -> int:
        """Count indexed chunks, optionally filtered by document_id."""
        if document_id:
            return len(self._doc_ids.get(document_id, set()))
        return len(self._chunks)

    def clear(self) -> None:
        """Clear all indexed chunks and reset the index."""
        self._chunks.clear()
        self._doc_ids.clear()
        self._chunk_tokens.clear()
        self._chunk_term_counts.clear()
        self._chunk_lengths.clear()
        self._inverted_index.clear()

    def get_stats(self) -> BM25IndexStats:
        """Get index summary statistics."""
        total_chunks = len(self._chunks)
        avg_len = (
            sum(self._chunk_lengths.values()) / total_chunks
            if total_chunks > 0
            else 0.0
        )
        return BM25IndexStats(
            total_documents=len(self._doc_ids),
            total_chunks=total_chunks,
            avg_chunk_length=round(avg_len, 2),
            total_terms=len(self._inverted_index),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Execute BM25 Okapi lexical search over indexed chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id filter must be a non-empty string.")
            candidate_ids = self._doc_ids.get(document_id, set())
        else:
            candidate_ids = set(self._chunks.keys())

        if not candidate_ids:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Corpus statistics for candidate pool
        n_docs = len(candidate_ids)
        total_len = sum(self._chunk_lengths[cid] for cid in candidate_ids)
        avgdl = (total_len / n_docs) if n_docs > 0 else 1.0
        if avgdl == 0.0:
            avgdl = 1.0

        scores: dict[str, float] = defaultdict(float)

        for term in query_tokens:
            posting = self._inverted_index.get(term, set()) & candidate_ids
            n_term = len(posting)
            if n_term == 0:
                continue

            # Robertson-Zaragoza / Lucene non-negative smoothed IDF
            idf = math.log(1.0 + (n_docs - n_term + 0.5) / (n_term + 0.5))

            for cid in posting:
                tf = self._chunk_term_counts[cid][term]
                doc_len = self._chunk_lengths[cid]
                # Okapi BM25 TF normalization
                tf_norm = (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                )
                scores[cid] += idf * tf_norm

        # Filter by threshold and score > 0
        min_threshold = score_threshold if score_threshold is not None else 0.0
        matched_cids = [
            cid for cid, s in scores.items() if s > 0.0 and s >= min_threshold
        ]

        # Sort: score descending, then deterministic tiebreak (page_number asc, chunk_index asc)
        matched_cids.sort(
            key=lambda cid: (
                -scores[cid],
                self._chunks[cid].page_number,
                self._chunks[cid].chunk_index,
            )
        )

        results: list[RetrievedChunk] = []
        for rank, cid in enumerate(matched_cids[:top_k], start=1):
            chunk = self._chunks[cid]
            raw_s = float(scores[cid])
            results.append(
                RetrievedChunk(
                    rank=rank,
                    score=raw_s,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    metadata={"char_count": chunk.char_count},
                    retrieval_type="bm25",
                    raw_score=raw_s,
                )
            )

        return results


class BM25Retriever:
    """Service for sparse lexical BM25 retrieval over document chunks."""

    def __init__(
        self,
        index: BM25OkapiIndex | None = None,
        k1: float | None = None,
        b: float | None = None,
        epsilon: float | None = None,
    ) -> None:
        self.index = index or BM25OkapiIndex(k1=k1, b=b, epsilon=epsilon)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Retrieve top-k document chunks using BM25 lexical ranking."""
        target_top_k = (
            top_k if top_k is not None else settings.DEFAULT_RETRIEVAL_TOP_K
        )
        results = self.index.search(
            query=query,
            top_k=target_top_k,
            document_id=document_id,
            score_threshold=score_threshold,
        )
        return RetrievalResult(
            query=query,
            document_id=document_id,
            top_k=target_top_k,
            total_results=len(results),
            results=results,
        )

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Index chunks into the BM25 index."""
        return self.index.index_chunks(chunks)

    def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return self.index.delete_by_document(document_id)

    def count_chunks(self, document_id: str | None = None) -> int:
        """Count indexed chunks."""
        return self.index.count_chunks(document_id=document_id)

    def clear(self) -> None:
        """Clear the BM25 index."""
        self.index.clear()

    def get_stats(self) -> BM25IndexStats:
        """Get index statistics."""
        return self.index.get_stats()


bm25_retriever = BM25Retriever()
