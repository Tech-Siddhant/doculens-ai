import logging
import time
from fastapi import APIRouter, File, HTTPException, Path as FastApiPath, Query, Request, UploadFile, status

from app.core.config import settings
from app.core.ratelimit import check_rate_limit
from app.schemas.document import (
    ChunkingResult,
    DocumentListItem,
    DocumentUploadResponse,
    ExtractionResult,
)
from app.schemas.generation import (
    GenerationResult,
    QuestionRequest,
)
from app.schemas.retrieval import (
    HybridRetrievalQuery,
    HybridRetrievalResult,
    IndexingResult,
    RetrievalQuery,
    RetrievalResult,
    VisualRetrievalResult,
)
from app.services.bm25 import bm25_retriever
from app.services.hybrid import hybrid_retriever
from app.services.chunker import chunk_extraction_result
from app.services.embedder import embedding_service
from app.services.extractor import extract_text_and_metadata
from app.services.llm_provider import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.services.orchestrator import orchestrator
from app.services.retriever import retriever
from app.services.page_storage import get_stored_page_path
from app.services.storage import (
    generate_document_id,
    get_document_path,
    get_stored_document,
    is_valid_document_id,
    list_stored_documents,
    save_uploaded_pdf,
)
from fastapi.responses import FileResponse
from app.services.visual_embedder import visual_embedding_service
from app.services.visual_vector_store import visual_vector_store
from app.services.validator import PDFValidationError, validate_pdf_file
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


async def _read_upload_limited(file: UploadFile, max_size_bytes: int) -> bytes:
    """Read an upload in bounded chunks so oversized bodies are rejected early
    instead of buffered fully into memory first."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_size_bytes:
            max_mb = max_size_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File size exceeds maximum permitted limit of {max_mb:.1f} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _check_rate_limit(request: Request) -> None:
    """Reject over-limit requests with 429 when rate limiting is enabled."""
    if not settings.RATE_LIMIT_ENABLED:
        return
    client_key = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Too many requests. "
                f"Please wait {settings.RATE_LIMIT_WINDOW_SECONDS:.0f}s and retry."
            ),
        )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
) -> DocumentUploadResponse:
    t0 = time.perf_counter()
    content = await _read_upload_limited(file, settings.MAX_UPLOAD_SIZE_BYTES)

    try:
        validate_pdf_file(
            filename=file.filename,
            content=content,
            max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )
    except PDFValidationError as exc:
        logger.warning(
            "Document upload validation failed",
            extra={"operation": "upload", "error_type": "PDFValidationError", "status": "error",
                   "duration_ms": round((time.perf_counter() - t0) * 1000, 2)},
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    document_id = generate_document_id()
    save_uploaded_pdf(document_id, content)

    logger.info(
        "Document uploaded successfully",
        extra={"operation": "upload", "document_id": document_id, "status": "success",
               "duration_ms": round((time.perf_counter() - t0) * 1000, 2)},
    )
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown.pdf",
        content_type=file.content_type or "application/pdf",
        size_bytes=len(content),
        status="uploaded",
    )


@router.get("", response_model=list[DocumentListItem])
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of documents to return (1-200)"),
    offset: int = Query(default=0, ge=0, description="Offset index for pagination"),
) -> list[DocumentListItem]:
    """List all stored PDF documents in upload directory with pagination, sorted newest first."""
    docs = list_stored_documents()
    return docs[offset : offset + limit]


@router.get("/{document_id}", response_model=DocumentListItem)
async def get_document(document_id: str) -> DocumentListItem:
    """Retrieve metadata and status for a single document."""
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )
    doc = get_stored_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return doc



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
        result = extract_text_and_metadata(document_id, file_path)
        logger.info(
            "Document extraction completed",
            extra={"operation": "extraction", "document_id": document_id,
                   "stage": "extraction", "status": "success"},
        )
        return result
    except Exception as exc:
        logger.exception(
            "Extraction failed.",
            extra={"operation": "extraction", "document_id": document_id,
                   "stage": "extraction", "status": "error", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Extraction failed.",
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

    if chunk_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_size must be a positive integer.",
        )
    if chunk_size > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_size exceeds maximum permitted value of 10000.",
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be non-negative and strictly less than chunk_size.",
        )

    file_path = get_document_path(document_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    try:
        extraction_result = extract_text_and_metadata(document_id, file_path)
        result = chunk_extraction_result(
            extraction_result=extraction_result,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(
            "Document chunking completed",
            extra={"operation": "chunking", "document_id": document_id,
                   "stage": "chunking", "status": "success"},
        )
        return result
    except Exception as exc:
        logger.exception(
            "Chunking failed.",
            extra={"operation": "chunking", "document_id": document_id,
                   "stage": "chunking", "status": "error", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chunking failed.",
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

    if chunk_size <= 0 or chunk_size > 10000 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunking parameters: chunk_size must be between 1 and 10000 and chunk_overlap >= 0 and < chunk_size.",
        )

    file_path = get_document_path(document_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
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
        t0_idx = time.perf_counter()
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

        bm25_retriever.delete_by_document(document_id)
        if chunking_result.chunks:
            bm25_retriever.index_chunks(chunking_result.chunks)

        idx_dur = round((time.perf_counter() - t0_idx) * 1000, 2)
        from app.core.metrics import metrics_collector
        metrics_collector.record_ingestion(
            duration_ms=idx_dur,
            pages=len(extraction_result.pages),
            chunks=len(chunking_result.chunks),
        )

        logger.info(
            "Document indexing completed",
            extra={"operation": "indexing", "document_id": document_id,
                   "stage": "indexing", "status": "success",
                   "duration_ms": idx_dur},
        )
        return IndexingResult(
            document_id=document_id,
            total_pages=len(extraction_result.pages),
            total_chunks=len(chunking_result.chunks),
            total_embeddings=len(embedded_chunks),
            status="indexed",
        )
    except Exception as exc:
        logger.exception(
            "Indexing failed.",
            extra={"operation": "indexing", "document_id": document_id,
                   "stage": "indexing", "status": "error", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indexing failed.",
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
        logger.exception(
            "Retrieval failed.",
            extra={"operation": "retrieval", "document_id": document_id,
                   "stage": "retrieval", "status": "error", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval failed.",
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
        logger.exception("Retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval failed.",
        ) from exc


@router.get("/{document_id}/pages/{page_number}/image", response_class=FileResponse)
async def get_page_image(document_id: str, page_number: int) -> FileResponse:
    """Serve the rendered page image."""
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )
    if page_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page_number must be >= 1",
        )
    if page_number > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page_number exceeds maximum permitted value of 5000.",
        )

    file_path = get_stored_page_path(document_id, page_number)
    if not file_path:
        doc_pdf_path = get_document_path(document_id)
        if doc_pdf_path and doc_pdf_path.is_file():
            try:
                from pathlib import Path
                from app.services.renderer import render_page
                rendered = render_page(document_id, doc_pdf_path, page_number)
                file_path = Path(rendered.image_path)
            except Exception:
                file_path = None

    if not file_path or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page_number} for document '{document_id}' not found.",
        )

    return FileResponse(
        path=file_path, 
        media_type="image/jpeg" if file_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png",
        filename=f"{document_id}_p{page_number}{file_path.suffix}"
    )


@router.post("/{document_id}/retrieve/visual", response_model=VisualRetrievalResult)
async def retrieve_visual(
    document_id: str,
    query_req: RetrievalQuery,
) -> VisualRetrievalResult:
    """Retrieve visually similar pages using a text query."""
    if not is_valid_document_id(document_id):
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
        # Check if doc exists or has visual pages indexed
        pages_count = visual_vector_store.count_pages(document_id)
        if pages_count == 0:
            file_path = get_document_path(document_id)
            if not file_path:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{document_id}' not found.",
                )
            
        # 1. Embed textual query into visual multi-modal space
        query_vector = visual_embedding_service.embed_visual_query(query_req.query)
        if not query_vector:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate visual query embedding.",
            )

        # 2. Search visual vector store
        results = visual_vector_store.search_visual(
            query_vector=query_vector,
            top_k=query_req.top_k,
            document_id=document_id,
            score_threshold=query_req.score_threshold,
        )

        return VisualRetrievalResult(
            query=query_req.query,
            document_id=document_id,
            top_k=query_req.top_k,
            total_results=len(results),
            results=results,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Visual retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Visual retrieval failed.",
        ) from exc


@router.post("/{document_id}/retrieve/bm25", response_model=RetrievalResult)
@router.post("/{document_id}/query-bm25", response_model=RetrievalResult)
async def retrieve_bm25_document(
    document_id: str,
    query_req: RetrievalQuery,
) -> RetrievalResult:
    """Retrieve document chunks using BM25 sparse lexical retrieval for a document."""
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    chunks_count = bm25_retriever.count_chunks(document_id)
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
        return bm25_retriever.retrieve(
            query=query_req.query,
            top_k=query_req.top_k,
            document_id=document_id,
            score_threshold=query_req.score_threshold,
        )
    except Exception as exc:
        logger.exception("BM25 retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BM25 retrieval failed.",
        ) from exc


@router.post("/retrieve/bm25", response_model=RetrievalResult)
@router.post("/query-bm25", response_model=RetrievalResult)
async def retrieve_bm25_collection(
    query_req: RetrievalQuery,
    document_id: str | None = Query(default=None, description="Optional document ID filter"),
) -> RetrievalResult:
    """Retrieve document chunks using BM25 sparse lexical retrieval across documents."""
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
        return bm25_retriever.retrieve(
            query=query_req.query,
            top_k=query_req.top_k,
            document_id=document_id,
            score_threshold=query_req.score_threshold,
        )
    except Exception as exc:
        logger.exception("BM25 retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BM25 retrieval failed.",
        ) from exc


@router.post("/{document_id}/retrieve/hybrid", response_model=HybridRetrievalResult)
@router.post("/{document_id}/query-hybrid", response_model=HybridRetrievalResult)
async def retrieve_hybrid_document(
    document_id: str,
    query_req: HybridRetrievalQuery,
) -> HybridRetrievalResult:
    """Retrieve fused multi-channel candidates (dense, BM25, visual) for a document."""
    if not is_valid_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document identifier format: '{document_id}'",
        )

    file_path = get_document_path(document_id)
    chunks_count = vector_store.count_chunks(document_id)
    bm25_count = bm25_retriever.count_chunks(document_id)
    visual_count = visual_vector_store.count_pages(document_id)
    if not file_path and chunks_count == 0 and bm25_count == 0 and visual_count == 0:
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
        return hybrid_retriever.retrieve(
            query=query_req.query,
            top_k=query_req.top_k,
            document_id=document_id,
            strategy=query_req.strategy,
            weights=query_req.weights,
            rrf_k=query_req.rrf_k,
            score_threshold=query_req.score_threshold,
            include_dense=query_req.include_dense,
            include_bm25=query_req.include_bm25,
            include_visual=query_req.include_visual,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Hybrid retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid retrieval failed.",
        ) from exc


@router.post("/retrieve/hybrid", response_model=HybridRetrievalResult)
@router.post("/query-hybrid", response_model=HybridRetrievalResult)
async def retrieve_hybrid_collection(
    query_req: HybridRetrievalQuery,
    document_id: str | None = Query(default=None, description="Optional document ID filter"),
) -> HybridRetrievalResult:
    """Retrieve fused multi-channel candidates across documents."""
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
        return hybrid_retriever.retrieve(
            query=query_req.query,
            top_k=query_req.top_k,
            document_id=document_id,
            strategy=query_req.strategy,
            weights=query_req.weights,
            rrf_k=query_req.rrf_k,
            score_threshold=query_req.score_threshold,
            include_dense=query_req.include_dense,
            include_bm25=query_req.include_bm25,
            include_visual=query_req.include_visual,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Hybrid retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid retrieval failed.",
        ) from exc


@router.post("/{document_id}/ask", response_model=GenerationResult)
async def ask_document(
    document_id: str,
    ask_req: QuestionRequest,
    request: Request,
) -> GenerationResult:
    """Retrieve dense evidence and generate a grounded answer for a document."""
    _check_rate_limit(request)

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
        t0_ask = time.perf_counter()
        result = orchestrator.orchestrate_query(
            question=ask_req.question,
            document_id=document_id,
            top_k=ask_req.top_k,
            score_threshold=ask_req.score_threshold,
            answer_style=ask_req.answer_style,
        )
        logger.info(
            "Answer generation completed",
            extra={"operation": "generation", "document_id": document_id,
                   "stage": "generation", "status": "success",
                   "duration_ms": round((time.perf_counter() - t0_ask) * 1000, 2)},
        )
        return result
    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Answer generation timed out while waiting for LLM provider response.",
        ) from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM provider rate limit exceeded. Please retry after waiting.",
        ) from exc
    except ProviderAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider authentication failed. Please check API key configuration.",
        ) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is temporarily unavailable. Please retry later.",
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider returned an invalid or malformed response.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Answer generation failed.",
            extra={"operation": "generation", "document_id": document_id,
                   "stage": "generation", "status": "error", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generation failed.",
        ) from exc


@router.post("/ask", response_model=GenerationResult)
async def ask_collection(
    ask_req: QuestionRequest,
    request: Request,
    document_id: str | None = Query(default=None, description="Optional document ID filter"),
) -> GenerationResult:
    """Retrieve dense evidence and generate a grounded answer across documents."""
    _check_rate_limit(request)

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
        t0_ask = time.perf_counter()
        result = orchestrator.orchestrate_query(
            question=ask_req.question,
            document_id=document_id,
            top_k=ask_req.top_k,
            score_threshold=ask_req.score_threshold,
            answer_style=ask_req.answer_style,
        )
        logger.info(
            "Answer generation completed",
            extra={"operation": "generation", "document_id": document_id,
                   "stage": "generation", "status": "success",
                   "duration_ms": round((time.perf_counter() - t0_ask) * 1000, 2)},
        )
        return result
    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Answer generation timed out while waiting for LLM provider response.",
        ) from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM provider rate limit exceeded. Please retry after waiting.",
        ) from exc
    except ProviderAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider authentication failed. Please check API key configuration.",
        ) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is temporarily unavailable. Please retry later.",
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider returned an invalid or malformed response.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Answer generation failed.",
            extra={"operation": "generation", "document_id": document_id,
                   "stage": "generation", "status": "error", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generation failed.",
        ) from exc
