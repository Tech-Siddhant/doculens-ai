"""Core business services for document processing."""

from app.services.bm25 import (
    BM25OkapiIndex,
    BM25Retriever,
    bm25_retriever,
    tokenize,
)
from app.services.chunker import chunk_extraction_result, chunk_text
from app.services.citation_validator import (
    CitationBuilder,
    CitationValidator,
    citation_validator,
    validate_citations,
)
from app.services.embedder import EmbeddingService, embedding_service
from app.services.extractor import extract_text_and_metadata, normalize_whitespace
from app.services.generator import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AnswerGenerator,
    build_grounding_prompt,
    generator,
)
from app.services.hybrid import (
    HybridRetriever,
    collect_and_merge_candidates,
    fuse_rrf,
    fuse_weighted,
    get_candidate_key,
    hybrid_retriever,
)
from app.services.llm_provider import (
    BaseLLMProvider,
    GoogleGeminiProvider,
    LLMResponse,
    MockLLMProvider,
    OpenAICompatibleProvider,
    get_llm_provider,
    llm_provider,
)
from app.services.normalizer import (
    min_max_scale_scores,
    normalize_retrieval_results,
)
from app.services.reranker import (
    CrossEncoderReranker,
    reranker,
)
from app.services.retriever import DenseRetriever, retriever
from app.services.storage import (
    generate_document_id,
    get_document_path,
    get_upload_dir,
    is_valid_document_id,
    save_uploaded_pdf,
)
from app.services.validator import PDFValidationError, validate_pdf_file
from app.services.vector_store import (
    QdrantVectorStore,
    chunk_id_to_point_id,
    vector_store,
)

__all__ = [
    "PDFValidationError",
    "validate_pdf_file",
    "generate_document_id",
    "is_valid_document_id",
    "save_uploaded_pdf",
    "get_document_path",
    "get_upload_dir",
    "normalize_whitespace",
    "extract_text_and_metadata",
    "chunk_text",
    "chunk_extraction_result",
    "EmbeddingService",
    "embedding_service",
    "chunk_id_to_point_id",
    "QdrantVectorStore",
    "vector_store",
    "DenseRetriever",
    "retriever",
    "BM25OkapiIndex",
    "BM25Retriever",
    "bm25_retriever",
    "tokenize",
    "min_max_scale_scores",
    "normalize_retrieval_results",
    "HybridRetriever",
    "hybrid_retriever",
    "fuse_weighted",
    "fuse_rrf",
    "collect_and_merge_candidates",
    "get_candidate_key",
    "BaseLLMProvider",
    "GoogleGeminiProvider",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "get_llm_provider",
    "llm_provider",
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "build_grounding_prompt",
    "AnswerGenerator",
    "generator",
    "CitationValidator",
    "CitationBuilder",
    "citation_validator",
    "validate_citations",
    "CrossEncoderReranker",
    "reranker",
]

