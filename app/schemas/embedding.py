from pydantic import BaseModel, Field


class EmbeddedChunk(BaseModel):
    chunk_id: str = Field(..., description="Deterministic ID formatted as {document_id}_p{page_number}_c{chunk_index}")
    document_id: str
    page_number: int = Field(..., description="1-based page number")
    chunk_index: int = Field(..., description="0-based chunk index within page")
    text: str
    embedding: list[float] = Field(..., description="Dense vector embedding values")
    dimension: int = Field(..., description="Dimensionality of the embedding vector")


class EmbeddingResult(BaseModel):
    document_id: str
    model_name: str
    dimension: int
    total_embeddings: int
    chunks: list[EmbeddedChunk]


class QueryEmbedding(BaseModel):
    query: str
    model_name: str
    dimension: int
    embedding: list[float] = Field(..., description="Dense vector embedding for the query")

class VisualPageEmbedding(BaseModel):
    document_id: str
    page_number: int = Field(..., description="1-based page number")
    embedding: list[float] = Field(..., description="Dense vector vision embedding")
    dimension: int


class VisualEmbeddingResult(BaseModel):
    document_id: str
    model_name: str
    dimension: int
    total_embeddings: int
    pages: list[VisualPageEmbedding]

