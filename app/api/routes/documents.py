from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from app.core.config import settings
from app.ingestion.chunker import chunk_document
from app.ingestion.extractor import extract_text_and_metadata
from app.ingestion.storage import get_pdf_path, save_uploaded_pdf
from app.ingestion.validator import PDFValidationError, validate_pdf_file
from app.schemas.documents import ChunkingResult, DocumentUploadResponse, ExtractionResult

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accepts a PDF document upload, validates its structure, and saves it to storage."""
    filename = file.filename or "uploaded.pdf"
    content = await file.read()

    try:
        validate_pdf_file(
            content=content,
            filename=filename,
            max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )
    except PDFValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    document_id, _ = save_uploaded_pdf(
        content=content,
        filename=filename,
        storage_dir=settings.STORAGE_DIR,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_size_bytes=len(content),
        status="uploaded",
        message="Document successfully uploaded and validated.",
    )


@router.post("/documents/{document_id}/extract", response_model=ExtractionResult)
def extract_document(document_id: str) -> ExtractionResult:
    """Extracts text by page and metadata for an uploaded document ID."""
    try:
        file_path = get_pdf_path(document_id, settings.STORAGE_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        return extract_text_and_metadata(file_path, document_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/documents/{document_id}/chunk", response_model=ChunkingResult)
def chunk_extracted_document(
    document_id: str,
    chunk_size: int = Query(500, ge=50, le=5000, description="Maximum characters per chunk"),
    chunk_overlap: int = Query(50, ge=0, description="Character overlap between chunks"),
) -> ChunkingResult:
    """Extracts and chunks an uploaded document's text into deterministic page-aware chunks."""
    try:
        file_path = get_pdf_path(document_id, settings.STORAGE_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        extraction_result = extract_text_and_metadata(file_path, document_id)
        return chunk_document(
            extraction_result=extraction_result,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from app.core.config import settings
from app.ingestion.chunker import chunk_document
from app.ingestion.extractor import extract_text_and_metadata
from app.ingestion.storage import get_pdf_path, save_uploaded_pdf
from app.ingestion.validator import PDFValidationError, validate_pdf_file
from app.schemas.documents import ChunkingResult, DocumentUploadResponse, ExtractionResult

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accepts a PDF document upload, validates its structure, and saves it to storage."""
    filename = file.filename or "uploaded.pdf"
    content = await file.read()

    try:
        validate_pdf_file(
            content=content,
            filename=filename,
            max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )
    except PDFValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    document_id, _ = save_uploaded_pdf(
        content=content,
        filename=filename,
        storage_dir=settings.STORAGE_DIR,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_size_bytes=len(content),
        status="uploaded",
        message="Document successfully uploaded and validated.",
    )


@router.post("/documents/{document_id}/extract", response_model=ExtractionResult)
def extract_document(document_id: str) -> ExtractionResult:
    """Extracts text by page and metadata for an uploaded document ID."""
    try:
        file_path = get_pdf_path(document_id, settings.STORAGE_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        return extract_text_and_metadata(file_path, document_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/documents/{document_id}/chunk", response_model=ChunkingResult)
def chunk_extracted_document(
    document_id: str,
    chunk_size: int = Query(500, ge=50, le=5000, description="Maximum characters per chunk"),
    chunk_overlap: int = Query(50, ge=0, description="Character overlap between chunks"),
) -> ChunkingResult:
    """Extracts and chunks an uploaded document's text into deterministic page-aware chunks."""
    try:
        file_path = get_pdf_path(document_id, settings.STORAGE_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        extraction_result = extract_text_and_metadata(file_path, document_id)
        return chunk_document(
            extraction_result=extraction_result,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
