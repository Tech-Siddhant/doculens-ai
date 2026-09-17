from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size_bytes: int
    status: str = "uploaded"
    message: str = "Document successfully uploaded and validated."


class ExtractedPage(BaseModel):
    page_number: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Raw extracted text from page")
    character_count: int = Field(..., description="Total characters in extracted text")


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    total_pages: int = 0


class ExtractionResult(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    pages: List[ExtractedPage]


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk ID (document_id_p{page}_c{idx})")
    document_id: str
    page_number: int = Field(..., description="1-based page number")
    chunk_index: int = Field(..., description="0-based chunk index on the page")
    text: str = Field(..., description="Chunk text snippet")
    character_count: int = Field(..., description="Length of chunk text")


class ChunkingResult(BaseModel):
    document_id: str
    total_chunks: int
    chunks: List[DocumentChunk]


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    content_type: str = "application/pdf"
    size_bytes: int
    status: str = "ready"
    total_pages: int = 0
    uploaded_at: Optional[str] = None

