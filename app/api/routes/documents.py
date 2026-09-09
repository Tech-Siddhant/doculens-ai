from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.core.config import settings
from app.schemas.document import (
    ChunkingResult,
    DocumentUploadResponse,
    ExtractionResult,
)
from app.schemas.generation import (
    GenerationResult,
    QuestionRequest,
)
from app.schemas.retrieval import (
    IndexingResult,
    RetrievalQuery,
    RetrievalResult,
)
from app.services.chunker import chunk_extraction_result
from app.services.embedder import embedding_service
from app.services.extractor import extract_text_and_metadata
from app.services.generator import generator
from app.services.retriever import retriever
from app.services.storage import (
    generate_document_id,
    get_document_path,
    is_valid_document_id,
    save_uploaded_pdf,
)
from app.services.validator import PDFValidationError, validate_pdf_file
from app.services.vector_store import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
) -> DocumentUploadResponse:
    content = await file.read()

    try:
        validate_pdf_file(
            filename=file.filename,
            content=content,
            max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )
    except PDFValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    document_id = generate_document_id()
    save_uploaded_pdf(document_id, content)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown.pdf",
        content_type=file.content_type or "application/pdf",
        size_bytes=len(content),
        status="uploaded",
    )


@router.post("/{document_id}/extract", response_model=ExtractionResult)
async def extract_document(document_id: str) -> ExtractionResult:
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    try:
        return extract_text_and_metadata(document_id, file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(exc)}",
        ) from exc


@router.post("/{document_id}/chunk", response_model=ChunkingResult)
async def chunk_document(
    document_id: str,
    chunk_size: int = Query(default=settings.DEFAULT_CHUNK_SIZE, description="Character chunk size"),
    chunk_overlap: int = Query(default=settings.DEFAULT_CHUNK_OVERLAP, description="Character chunk overlap"),
) -> ChunkingResult:
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if chunk_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_size must be a positive integer.",
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be non-negative and strictly less than chunk_size.",
        )

    try:
        extraction_result = extract_text_and_metadata(document_id, file_path)
        return chunk_extraction_result(
            extraction_result=extraction_result,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {str(exc)}",
        ) from exc


@router.post("/{document_id}/index", response_model=IndexingResult)
async def index_document(
    document_id: str,
    chunk_size: int = Query(default=settings.DEFAULT_CHUNK_SIZE, description="Character chunk size"),
    chunk_overlap: int = Query(default=settings.DEFAULT_CHUNK_OVERLAP, description="Character chunk overlap"),
) -> IndexingResult:
    """Extract, chunk, embed, and upsert document vectors into Qdrant."""
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunking parameters: chunk_size must be > 0 and chunk_overlap >= 0 and < chunk_size.",
        )

    try:
        extraction_result = extract_text_and_metadata(document_id, file_path)
        chunking_result = chunk_extraction_result(
            extraction_result=extraction_result,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embedded_chunks = embedding_service.embed_chunks(chunking_result.chunks)
        vector_store.delete_by_document(document_id)
        if embedded_chunks:
            vector_store.upsert_chunks(embedded_chunks)

        return IndexingResult(
            document_id=document_id,
            total_pages=len(extraction_result.pages),
            total_chunks=len(chunking_result.chunks),
            total_embeddings=len(embedded_chunks),
            status="indexed",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(exc)}",
        ) from exc


@router.post("/{document_id}/query", response_model=RetrievalResult)
async def query_document(
    document_id: str,
    query_req: RetrievalQuery,
) -> RetrievalResult:
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    chunks_count = vector_store.count_chunks(document_id)
    if not file_path and chunks_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if not query_req.query or not query_req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    if query_req.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer.",
        )

    try:
        return retriever.retrieve(
            query=query_req.query,
            top_k=query_req.top_k,
            document_id=document_id,
            score_threshold=query_req.score_threshold,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(exc)}",
        ) from exc


@router.post("/query", response_model=RetrievalResult)
async def query_collection(
    query_req: RetrievalQuery,
    document_id: str | None = Query(default=None, description="Optional document ID filter"),
) -> RetrievalResult:
    if document_id is not None and not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    if not query_req.query or not query_req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    if query_req.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer.",
        )

    try:
        return retriever.retrieve(
            query=query_req.query,
            top_k=query_req.top_k,
            document_id=document_id,
            score_threshold=query_req.score_threshold,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(exc)}",
        ) from exc


@router.post("/{document_id}/ask", response_model=GenerationResult)
async def ask_document(
    document_id: str,
    ask_req: QuestionRequest,
) -> GenerationResult:
    """Retrieve dense evidence and generate a grounded answer for a document."""
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    chunks_count = vector_store.count_chunks(document_id)
    if not file_path and chunks_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if not ask_req.question or not ask_req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )

    if ask_req.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer.",
        )

    try:
        retrieval_res = retriever.retrieve(
            query=ask_req.question,
            top_k=ask_req.top_k,
            document_id=document_id,
            score_threshold=ask_req.score_threshold,
        )
        return generator.generate_answer(
            question=ask_req.question,
            evidence=retrieval_res.results,
            document_id=document_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(exc)}",
        ) from exc


@router.post("/ask", response_model=GenerationResult)
async def ask_collection(
    ask_req: QuestionRequest,
    document_id: str | None = Query(default=None, description="Optional document ID filter"),
) -> GenerationResult:
    """Retrieve dense evidence and generate a grounded answer across documents."""
    if document_id is not None and not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    if not ask_req.question or not ask_req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )

    if ask_req.top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer.",
        )

    try:
        retrieval_res = retriever.retrieve(
            query=ask_req.question,
            top_k=ask_req.top_k,
            document_id=document_id,
            score_threshold=ask_req.score_threshold,
        )
        return generator.generate_answer(
            question=ask_req.question,
            evidence=retrieval_res.results,
            document_id=document_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(exc)}",
        ) from exc
