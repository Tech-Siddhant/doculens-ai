from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    rank: int = Field(..., ge=1, description="1-based retrieval ranking position")
    score: float = Field(..., description="Cosine similarity score")
    chunk_id: str = Field(..., description="Deterministic ID formatted as {document_id}_p{page_number}_c{chunk_index}")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_index: int = Field(..., ge=0, description="0-based chunk index within page")
    text: str = Field(..., description="Raw text content of the retrieved chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional chunk payload metadata")


class RetrievalQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, description="Maximum number of chunks to retrieve")
    score_threshold: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Optional minimum cosine similarity score threshold"
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
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional image metadata")


class VisualRetrievalResult(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int
    total_results: int
    results: list[RetrievedVisualPage]



class IndexingResult(BaseModel):
    document_id: str
    total_pages: int
    total_chunks: int
    total_embeddings: int
    status: str = "indexed"
