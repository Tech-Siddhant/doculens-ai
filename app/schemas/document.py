from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: str = "uploaded"


class ExtractedPage(BaseModel):
    page_number: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Extracted plain text")
    char_count: int = Field(..., description="Number of characters in extracted text")


class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    total_pages: int
    file_size_bytes: int


class ExtractionResult(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    pages: list[ExtractedPage]


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Deterministic ID formatted as {document_id}_p{page_number}_c{chunk_index}")
    document_id: str
    page_number: int = Field(..., description="1-based page number")
    chunk_index: int = Field(..., description="0-based chunk index within page")
    text: str
    char_count: int


class ChunkingResult(BaseModel):
    document_id: str
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[DocumentChunk]


class ChunkRequest(BaseModel):
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class RenderedPage(BaseModel):
    document_id: str
    page_number: int = Field(..., description="1-based page number")
    image_path: str = Field(..., description="Path to rendered page image file")
    width: int
    height: int
    format: str = Field(..., description="Image format: png or jpeg")
    size_bytes: int


class RenderingResult(BaseModel):
    document_id: str
    total_pages_rendered: int
    dpi: int
    format: str
    pages: list[RenderedPage]
