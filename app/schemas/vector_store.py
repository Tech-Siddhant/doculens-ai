from typing import Any

from pydantic import BaseModel, Field


class VectorPointRecord(BaseModel):
    point_id: str = Field(..., description="Deterministic UUIDv5 string derived from chunk_id")
    chunk_id: str = Field(..., description="Deterministic ID formatted as {document_id}_p{page_number}_c{chunk_index}")
    document_id: str
    page_number: int = Field(..., description="1-based page number")
    chunk_index: int = Field(..., description="0-based chunk index within page")
    text: str
    char_count: int
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorStoreStats(BaseModel):
    collection_name: str
    total_points: int
    dimension: int
    status: str = "ready"
