from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    rank: int = Field(..., ge=1, description="1-based retrieval ranking position")
    score: float = Field(..., description="Retrieval score (similarity or relevance)")
    chunk_id: str = Field(..., description="Deterministic ID formatted as {document_id}_p{page_number}_c{chunk_index}")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_index: int = Field(..., ge=0, description="0-based chunk index within page")
    text: str = Field(..., description="Raw text content of the retrieved chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional chunk payload metadata")
    retrieval_type: str = Field(default="dense", description="Retrieval method (dense, bm25, visual, hybrid)")
    raw_score: float | None = Field(default=None, description="Original unnormalized retrieval score")
    normalized_score: float | None = Field(default=None, description="Normalized score in [0.0, 1.0]")

    def model_post_init(self, __context: Any) -> None:
        if self.raw_score is None:
            self.raw_score = self.score


class RetrievalQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, description="Maximum number of items to retrieve")
    score_threshold: float | None = Field(
        default=None, description="Optional minimum retrieval score threshold"
    )


class RetrievalResult(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int
    total_results: int
    results: list[RetrievedChunk]


class RetrievedVisualPage(BaseModel):
    rank: int = Field(..., ge=1, description="1-based retrieval ranking position")
    score: float = Field(..., description="Cosine similarity score")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    image_url: str = Field(..., description="Relative API URL to retrieve rendered page image")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional image metadata")
    retrieval_type: str = Field(default="visual", description="Retrieval method")
    raw_score: float | None = Field(default=None, description="Original unnormalized retrieval score")
    normalized_score: float | None = Field(default=None, description="Normalized score in [0.0, 1.0]")

    def model_post_init(self, __context: Any) -> None:
        if self.raw_score is None:
            self.raw_score = self.score


class VisualRetrievalResult(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int
    total_results: int
    results: list[RetrievedVisualPage]


class BM25IndexStats(BaseModel):
    total_documents: int
    total_chunks: int
    avg_chunk_length: float
    total_terms: int


class ModalityWeights(BaseModel):
    dense: float = Field(default=0.5, ge=0.0, description="Weight for dense semantic retrieval")
    bm25: float = Field(default=0.3, ge=0.0, description="Weight for BM25 sparse retrieval")
    visual: float = Field(default=0.2, ge=0.0, description="Weight for visual page retrieval")

    def normalized_dict(self) -> dict[str, float]:
        """Return normalized weights summing to 1.0."""
        total = self.dense + self.bm25 + self.visual
        if total <= 0:
            raise ValueError("At least one modality weight must be strictly positive.")
        return {
            "dense": round(self.dense / total, 6),
            "bm25": round(self.bm25 / total, 6),
            "visual": round(self.visual / total, 6),
        }


class HybridRetrievalQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, description="Maximum number of items to retrieve")
    strategy: Literal["weighted", "rrf"] = Field(
        default="weighted", description="Fusion strategy: 'weighted' or 'rrf'"
    )
    weights: ModalityWeights | None = Field(
        default=None, description="Modality weights for weighted fusion"
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant k")
    score_threshold: float | None = Field(
        default=None, description="Optional minimum score threshold"
    )
    include_dense: bool = Field(
        default=True, description="Whether to include dense semantic retrieval channel"
    )
    include_bm25: bool = Field(
        default=True, description="Whether to include BM25 sparse retrieval channel"
    )
    include_visual: bool = Field(
        default=True, description="Whether to include visual retrieval channel"
    )


class FusedCandidate(BaseModel):
    rank: int = Field(..., ge=1, description="1-based retrieval ranking position")
    score: float = Field(..., description="Fused retrieval score")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_id: str | None = Field(default=None, description="Chunk ID for text chunk candidates")
    chunk_index: int | None = Field(default=None, description="0-based chunk index within page")
    text: str | None = Field(default=None, description="Raw text content if candidate has text")
    image_url: str | None = Field(default=None, description="Image URL if candidate has visual page")
    retrieval_type: str = Field(default="hybrid", description="Retrieval method (hybrid)")
    sources: list[str] = Field(..., description="Retrievers that returned this candidate (e.g. ['dense', 'bm25'])")
    raw_scores: dict[str, float] = Field(default_factory=dict, description="Raw unnormalized score per retriever")
    normalized_scores: dict[str, float] = Field(default_factory=dict, description="Normalized score per retriever")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Merged chunk/page metadata")


class HybridRetrievalResult(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int
    strategy: str
    total_results: int
    results: list[FusedCandidate]
    weights: dict[str, float] | None = None
    rrf_k: int | None = None


class IndexingResult(BaseModel):
    document_id: str
    total_pages: int
    total_chunks: int
    total_embeddings: int
    status: str = "indexed"


class RerankedCandidate(BaseModel):
    rank: int = Field(..., ge=1, description="1-based reranked position")
    score: float = Field(..., description="Cross-encoder relevance score")
    initial_rank: int = Field(..., ge=1, description="1-based rank prior to reranking")
    initial_score: float = Field(..., description="Retrieval/fusion score prior to reranking")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_id: str | None = Field(default=None, description="Chunk ID if text chunk candidate")
    chunk_index: int | None = Field(default=None, description="0-based chunk index within page")
    text: str | None = Field(default=None, description="Text content evaluated by reranker")
    image_url: str | None = Field(default=None, description="Rendered image URL if visual page")
    retrieval_type: str = Field(default="reranked", description="Retrieval method identifier")
    sources: list[str] = Field(default_factory=list, description="Retrievers that provided candidate")
    raw_scores: dict[str, float] = Field(default_factory=dict, description="Raw scores from earlier stages")
    normalized_scores: dict[str, float] = Field(default_factory=dict, description="Normalized scores from earlier stages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Payload and citation metadata")


class RerankResult(BaseModel):
    query: str
    document_id: str | None = None
    model_name: str
    top_k: int
    total_results: int
    results: list[RerankedCandidate]


class EvidenceItem(BaseModel):
    """Selected evidence item preserved with complete multi-modal provenance for context assembly."""

    rank: int = Field(..., ge=1, description="1-based selected evidence rank position")
    score: float = Field(..., description="Cross-encoder reranker relevance score")
    initial_rank: int = Field(..., ge=1, description="1-based rank prior to reranking")
    initial_score: float = Field(..., description="Retrieval/fusion score prior to reranking")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_id: str | None = Field(default=None, description="Chunk ID if text chunk candidate")
    chunk_index: int | None = Field(default=None, description="0-based chunk index within page")
    text: str | None = Field(default=None, description="Grounding text content for the evidence item")
    image_url: str | None = Field(default=None, description="Visual page image reference URL if applicable")
    retrieval_type: str = Field(default="evidence", description="Evidence classification identifier")
    sources: list[str] = Field(default_factory=list, description="Original retrieval channels (e.g. ['dense', 'bm25', 'visual'])")
    raw_scores: dict[str, float] = Field(default_factory=dict, description="Raw scores from retrievers")
    normalized_scores: dict[str, float] = Field(default_factory=dict, description="Normalized scores from retrievers")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Complete chunk/page payload and citation metadata")


class EvidenceSelectionResult(BaseModel):
    """Structured result of the top-K evidence selection stage."""

    query: str
    document_id: str | None = None
    top_k: int
    total_candidates: int
    total_selected: int
    evidence: list[EvidenceItem]


