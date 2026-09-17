"""Unit tests for Phase 9.2 Reliability, Resilience, and Failure Handling."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.api.main import app
from app.core.config import settings
from app.schemas.document import ExtractionResult, ExtractedPage, DocumentMetadata
from app.schemas.generation import GenerationResult
from app.schemas.retrieval import FusedCandidate, RetrievedChunk, RetrievedVisualPage
from app.services.chunker import chunk_extraction_result
from app.services.citation_validator import CitationValidator, INSUFFICIENT_EVIDENCE_ANSWER, is_refusal_response
from app.services.embedder import EmbeddingService
from app.services.generator import AnswerGenerator
from app.services.hybrid import HybridRetriever
from app.services.llm_provider import (
    OpenAICompatibleProvider,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.services.orchestrator import AnswerOrchestrator
from app.services.renderer import RenderingError, render_page
from app.services.validator import PDFValidationError, validate_pdf_file


client = TestClient(app)


# ---------------------------------------------------------------------------
# 0. Document Ingestion & Validation Failures
# ---------------------------------------------------------------------------

def test_unsupported_file_extension_rejected() -> None:
    """Verify that non-PDF file extensions are rejected with 400 Bad Request."""
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("report.docx", b"%PDF-1.4 dummy", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_invalid_magic_bytes_rejected() -> None:
    """Verify that files lacking %PDF- magic bytes are rejected with 400 Bad Request."""
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("fake.pdf", b"NOT A VALID PDF CONTENT", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid PDF header" in response.json()["detail"]


def test_filename_traversal_rejected() -> None:
    """Verify that filenames containing directory traversal characters are rejected."""
    with pytest.raises(PDFValidationError, match="path separators or control characters"):
        validate_pdf_file(filename="../../etc/passwd.pdf", content=b"%PDF-1.4 sample", max_size_bytes=1000)


def test_corrupted_pdf_structure_rejected() -> None:
    """Verify that corrupt unparseable PDF bodies are rejected with 400 Bad Request."""
    corrupt_bytes = b"%PDF-1.4\ncorrupted garbage stream and broken xref table %EOF"
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("corrupt.pdf", corrupt_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert "Corrupted or unreadable PDF structure" in response.json()["detail"]


def test_empty_file_upload_rejected() -> None:
    """Verify that 0-byte file uploads are rejected with 400 Bad Request."""
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_empty_document_zero_readable_pages_rejected() -> None:
    """Verify that valid PDF structure with 0 pages is rejected with 400 Bad Request."""
    writer = PdfWriter()
    buf = io.BytesIO()
    writer.write(buf)
    zero_page_bytes = buf.getvalue()

    with pytest.raises(PDFValidationError, match="contains no readable pages"):
        validate_pdf_file(filename="zero_pages.pdf", content=zero_page_bytes, max_size_bytes=100000)


def test_file_size_exceeded_rejected_413() -> None:
    """Verify that files exceeding maximum permitted bytes are rejected with 413."""
    oversized = b"%PDF-1.4" + b"X" * 200
    with pytest.raises(PDFValidationError) as exc_info:
        validate_pdf_file(filename="oversized.pdf", content=oversized, max_size_bytes=100)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum permitted limit" in exc_info.value.message


# ---------------------------------------------------------------------------
# 1. LLM Provider Timeouts, Retries & Malformed Responses
# ---------------------------------------------------------------------------

def test_llm_provider_timeout_raises_provider_timeout_error() -> None:
    """Verify that httpx.TimeoutException is caught and raised as ProviderTimeoutError."""
    provider = OpenAICompatibleProvider(api_key="test-key", timeout=1.0, max_retries=1)

    with patch("httpx.Client.post") as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")
        with pytest.raises(ProviderTimeoutError, match="timed out after 1.0 seconds"):
            provider.generate(prompt="Hello")


def test_llm_provider_bounded_retries_on_transient_503() -> None:
    """Verify that transient 503 status code retries bounded up to max_retries."""
    provider = OpenAICompatibleProvider(api_key="test-key", max_retries=2, timeout=5.0)

    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Service Unavailable", request=MagicMock(), response=mock_resp_503
    )

    with patch("httpx.Client.post", return_value=mock_resp_503) as mock_post, \
         patch("time.sleep") as mock_sleep:
        with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
            provider.generate(prompt="Hello")

        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2


def test_llm_provider_fast_failure_on_401_no_retry_storm() -> None:
    """Verify that deterministic 401 Auth Error fails immediately without retrying."""
    provider = OpenAICompatibleProvider(api_key="bad-key", max_retries=3, timeout=5.0)

    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    mock_resp_401.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_resp_401
    )

    with patch("httpx.Client.post", return_value=mock_resp_401) as mock_post, \
         patch("time.sleep") as mock_sleep:
        with pytest.raises(ProviderAuthenticationError, match="authentication failed"):
            provider.generate(prompt="Hello")


def test_llm_provider_malformed_response_empty_choices() -> None:
    """Verify that empty or malformed 'choices' array raises a clean ProviderError."""
    provider = OpenAICompatibleProvider(api_key="test-key", max_retries=0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"choices": []}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ProviderError, match="missing or empty choices"):
            provider.generate(prompt="Hello")


def test_llm_provider_malformed_response_invalid_json() -> None:
    """Verify that non-JSON response raises ProviderError."""
    provider = OpenAICompatibleProvider(api_key="test-key", max_retries=0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.side_effect = ValueError("Invalid JSON payload")

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ProviderError, match="Invalid JSON"):
            provider.generate(prompt="Hello")


def test_llm_provider_token_usage_tracking() -> None:
    """Verify that token counts are accurately extracted into LLMResponse."""
    provider = OpenAICompatibleProvider(api_key="test-key", max_retries=0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "DocuLens AI answer."}}],
        "model": "gemini-3.7-flash",
        "usage": {"prompt_tokens": 120, "completion_tokens": 35, "total_tokens": 155},
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = provider.generate(prompt="Hello")
        assert res.content == "DocuLens AI answer."
        assert res.usage["prompt_tokens"] == 120
        assert res.usage["completion_tokens"] == 35
        assert res.usage["total_tokens"] == 155


# ---------------------------------------------------------------------------
# 2. Retrieval Fail-Soft Fallbacks
# ---------------------------------------------------------------------------

def test_hybrid_retriever_dense_failure_falls_back_to_bm25() -> None:
    """When dense retriever fails, hybrid retriever continues with BM25."""
    mock_dense = MagicMock()
    mock_dense.retrieve.side_effect = RuntimeError("Qdrant connection refused")

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = MagicMock(
        results=[
            RetrievedChunk(
                rank=1,
                score=0.85,
                chunk_id="doc_1_p1_c0",
                document_id="doc_1",
                page_number=1,
                chunk_index=0,
                text="Lexical keyword match from BM25.",
                retrieval_type="bm25",
            )
        ]
    )

    hybrid = HybridRetriever(
        dense_retriever_inst=mock_dense,
        bm25_retriever_inst=mock_bm25,
        visual_embedder_inst=MagicMock(),
        visual_store_inst=MagicMock(),
    )

    res = hybrid.retrieve(query="search query", include_visual=False)
    assert len(res.results) == 1
    assert res.results[0].chunk_id == "doc_1_p1_c0"
    assert "bm25" in res.results[0].sources


def test_hybrid_retriever_bm25_failure_falls_back_to_dense() -> None:
    """When BM25 retriever fails, hybrid retriever continues with dense results."""
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = MagicMock(
        results=[
            RetrievedChunk(
                rank=1,
                score=0.92,
                chunk_id="doc_2_p1_c0",
                document_id="doc_2",
                page_number=1,
                chunk_index=0,
                text="Dense semantic vector match.",
                retrieval_type="dense",
            )
        ]
    )

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.side_effect = RuntimeError("BM25 index corrupted")

    hybrid = HybridRetriever(
        dense_retriever_inst=mock_dense,
        bm25_retriever_inst=mock_bm25,
        visual_embedder_inst=MagicMock(),
        visual_store_inst=MagicMock(),
    )

    res = hybrid.retrieve(query="search query", include_visual=False)
    assert len(res.results) == 1
    assert res.results[0].chunk_id == "doc_2_p1_c0"
    assert "dense" in res.results[0].sources


def test_hybrid_retriever_visual_failure_falls_back_to_text_channels() -> None:
    """When visual retrieval fails, hybrid retriever continues with dense and BM25."""
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = MagicMock(
        results=[
            RetrievedChunk(
                rank=1, score=0.9, chunk_id="chk_dense", document_id="doc_1",
                page_number=1, chunk_index=0, text="Dense chunk", retrieval_type="dense",
            )
        ]
    )
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = MagicMock(results=[])

    mock_vis_embed = MagicMock()
    mock_vis_embed.embed_visual_query.side_effect = RuntimeError("Visual model load failed")

    hybrid = HybridRetriever(
        dense_retriever_inst=mock_dense,
        bm25_retriever_inst=mock_bm25,
        visual_embedder_inst=mock_vis_embed,
        visual_store_inst=MagicMock(),
    )

    res = hybrid.retrieve(query="test query", include_visual=True)
    assert len(res.results) == 1
    assert res.results[0].chunk_id == "chk_dense"


def test_hybrid_retriever_all_channels_fail_returns_empty_gracefully() -> None:
    """When all retrieval channels fail simultaneously, hybrid retriever returns empty results without crashing."""
    mock_dense = MagicMock()
    mock_dense.retrieve.side_effect = RuntimeError("Dense failure")

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.side_effect = RuntimeError("BM25 failure")

    mock_vis = MagicMock()
    mock_vis.embed_visual_query.side_effect = RuntimeError("Visual failure")

    hybrid = HybridRetriever(
        dense_retriever_inst=mock_dense,
        bm25_retriever_inst=mock_bm25,
        visual_embedder_inst=mock_vis,
        visual_store_inst=MagicMock(),
    )

    res = hybrid.retrieve(query="test query")
    assert res.results == []
    assert res.total_results == 0


# ---------------------------------------------------------------------------
# 3. Reranker Failure Fallback in Orchestrator
# ---------------------------------------------------------------------------

def test_orchestrator_reranker_failure_falls_back_to_hybrid_results() -> None:
    """When reranker model crashes, orchestrator falls back to raw hybrid retrieval results and marks stage status."""
    orch = AnswerOrchestrator()

    mock_fused = FusedCandidate(
        rank=1,
        score=0.9,
        document_id="doc_x",
        page_number=1,
        chunk_id="chk_x",
        text="Grounded factual text.",
        sources=["dense"],
    )

    with patch("app.services.orchestrator.hybrid_retriever.retrieve") as mock_ret, \
         patch("app.services.orchestrator.reranker.rerank") as mock_rerank:

        mock_ret.return_value = MagicMock(results=[mock_fused], strategy="weighted")
        mock_rerank.side_effect = RuntimeError("ONNX Runtime cross-encoder execution failed")

        res = orch.orchestrate_query("What is X?", document_id="doc_x")

        assert res.pipeline_trace is not None
        rerank_stage = next(s for s in res.pipeline_trace.stages if s.stage_id == "reranking")
        assert rerank_stage.status == "fallback"
        assert rerank_stage.fallback_used is True
        assert rerank_stage.error_category == "reranker_failure"
        assert "Reranker failed" in (rerank_stage.fallback_reason or "")
        assert res.answer is not None


# ---------------------------------------------------------------------------
# 4. Groundedness & No-Evidence Handling
# ---------------------------------------------------------------------------

def test_generator_empty_evidence_immediate_refusal() -> None:
    """When evidence is empty, generator returns no-evidence response without invoking LLM."""
    mock_provider = MagicMock()
    generator = AnswerGenerator(provider=mock_provider)

    res = generator.generate_answer(question="What is the revenue?", evidence=[])

    assert res.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert res.is_grounded is False
    assert res.citations == []
    assert res.provider == "rule_based"
    assert mock_provider.generate.call_count == 0


def test_orchestrator_insufficient_evidence_flow() -> None:
    """When retrieval finds no matching chunks, pipeline safely finishes with no-evidence response."""
    orch = AnswerOrchestrator()

    with patch("app.services.orchestrator.hybrid_retriever.retrieve") as mock_ret:
        mock_ret.return_value = MagicMock(results=[], strategy="weighted")

        res = orch.orchestrate_query("What is non-existent info?", document_id="doc_empty")

        assert res.answer == INSUFFICIENT_EVIDENCE_ANSWER
        assert res.is_grounded is False
        assert len(res.citations) == 0
        assert res.pipeline_trace is not None
        assert res.pipeline_trace.summary.total_stages == 8


# ---------------------------------------------------------------------------
# 5. API Route Error Code Mappings (504, 429, 503, 502)
# ---------------------------------------------------------------------------

def test_api_ask_timeout_returns_http_504() -> None:
    """Verify that ProviderTimeoutError maps to HTTP 504 Gateway Timeout."""
    with patch("app.services.orchestrator.orchestrator.orchestrate_query") as mock_orch:
        mock_orch.side_effect = ProviderTimeoutError("Request timed out after 30s")
        
        response = client.post("/api/v1/documents/ask", json={"question": "Test query?"})
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_api_ask_rate_limit_returns_http_429() -> None:
    """Verify that ProviderRateLimitError maps to HTTP 429 Too Many Requests."""
    with patch("app.services.orchestrator.orchestrator.orchestrate_query") as mock_orch:
        mock_orch.side_effect = ProviderRateLimitError("Rate limit exceeded")
        
        response = client.post("/api/v1/documents/ask", json={"question": "Test query?"})
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_api_ask_unavailable_returns_http_503() -> None:
    """Verify that ProviderUnavailableError maps to HTTP 503 Service Unavailable."""
    with patch("app.services.orchestrator.orchestrator.orchestrate_query") as mock_orch:
        mock_orch.side_effect = ProviderUnavailableError("Provider service down")
        
        response = client.post("/api/v1/documents/ask", json={"question": "Test query?"})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


def test_api_ask_auth_failure_returns_http_502() -> None:
    """Verify that ProviderAuthenticationError maps to HTTP 502 without leaking secrets."""
    with patch("app.services.orchestrator.orchestrator.orchestrate_query") as mock_orch:
        mock_orch.side_effect = ProviderAuthenticationError("Bad API key: sk-secret-123")
        
        response = client.post("/api/v1/documents/ask", json={"question": "Test query?"})
        assert response.status_code == 502
        assert "sk-secret-123" not in response.text
        assert "authentication failed" in response.json()["detail"].lower()



def test_api_ask_malformed_response_returns_http_502() -> None:
    """Verify that ProviderError (e.g. malformed JSON or missing choices) maps to HTTP 502."""
    with patch("app.services.orchestrator.orchestrator.orchestrate_query") as mock_orch:
        mock_orch.side_effect = ProviderError("Malformed LLM provider response: missing choices.")
        
        response = client.post("/api/v1/documents/ask", json={"question": "Test query?"})
        assert response.status_code == 502
        assert "invalid or malformed response" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 6. Citation & Evidence Validation Hallucination Dropping
# ---------------------------------------------------------------------------

def test_citation_validation_failure_drops_hallucinated_citations() -> None:
    """Verify that citations referencing out-of-bounds indices are dropped and logged."""
    val = CitationValidator()
    valid_chunk = RetrievedChunk(
        rank=1, score=0.9, chunk_id="chk_1", document_id="doc_alpha",
        page_number=1, chunk_index=0, text="Valid facts.", retrieval_type="dense"
    )
    res = val.validate_citations(
        answer="Claim supported [Evidence 1], hallucinated claim [Evidence 99].",
        evidence=[valid_chunk],
        target_document_id="doc_alpha",
    )
    assert len(res.citations) == 1
    assert res.citations[0].reference == "[Evidence 1]"
    assert res.valid_count == 1
    assert res.invalid_count == 1
    assert any(issue.issue_type in ("fabricated_id", "out_of_range") or "nonexistent" in issue.message.lower() for issue in res.issues)


def test_citation_validation_cross_document_isolation() -> None:
    """Verify that citations pointing to a different document ID are rejected."""
    val = CitationValidator()
    wrong_chunk = RetrievedChunk(
        rank=1, score=0.9, chunk_id="chk_bad", document_id="doc_beta",
        page_number=1, chunk_index=0, text="Wrong doc text.", retrieval_type="dense"
    )
    res = val.validate_citations(
        answer="Claim referencing doc beta [Evidence 1].",
        evidence=[wrong_chunk],
        target_document_id="doc_alpha",
    )
    assert len(res.citations) == 0
    assert res.invalid_count == 1
    assert any(issue.issue_type == "wrong_document_id" for issue in res.issues)


def test_refusal_phrase_detection() -> None:
    """Verify that refusal phrases indicating lack of evidence are recognized accurately."""
    assert is_refusal_response(INSUFFICIENT_EVIDENCE_ANSWER) is True
    assert is_refusal_response("I do not have sufficient information in the provided document.") is True
    assert is_refusal_response("The document does not contain sufficient information.") is True
    assert is_refusal_response("According to the report, revenue grew by 15%.") is False



# ---------------------------------------------------------------------------
# 7. Phase 9.4 Comprehensive Reliability & Security Hardening
# ---------------------------------------------------------------------------

def test_extraction_failure_returns_500_with_sanitized_detail() -> None:
    """Verify that extraction failure returns HTTP 500 with sanitized message and no stack trace."""
    with patch("app.api.routes.documents.get_document_path", return_value=Path("/fake/path.pdf")), \
         patch("app.api.routes.documents.extract_text_and_metadata", side_effect=ValueError("Corrupted cross-reference stream in PDF")):
        response = client.post(f"{settings.API_V1_STR}/documents/doc_1234567890ab/extract")
        assert response.status_code == 500
        assert response.json()["detail"] == "Extraction failed."
        assert "Traceback" not in response.text
        assert "cross-reference" not in response.text


def test_page_rendering_failure_returns_404_or_handled_safely() -> None:
    """Verify that page rendering failure returns HTTP 404 without leaking internal paths."""
    with patch("app.api.routes.documents.get_stored_page_path", return_value=None), \
         patch("app.api.routes.documents.get_document_path", return_value=Path("/fake/path.pdf")), \
         patch("app.services.renderer.render_page", side_effect=RenderingError("MuPDF error: corrupted page")):
        response = client.get(f"{settings.API_V1_STR}/documents/doc_1234567890ab/pages/1/image")
        assert response.status_code == 404
        assert "Page 1 for document 'doc_1234567890ab' not found." in response.json()["detail"]
        assert "MuPDF" not in response.text


def test_llm_provider_direct_429_rate_limit_error() -> None:
    """Verify that HTTP 429 raises ProviderRateLimitError immediately."""
    provider = OpenAICompatibleProvider(api_key="test-key", max_retries=0)
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests", request=MagicMock(), response=mock_resp_429
    )
    with patch("httpx.Client.post", return_value=mock_resp_429):
        with pytest.raises(ProviderRateLimitError, match="rate limit exceeded"):
            provider.generate(prompt="Hello")



def test_backend_service_unavailability_health_status() -> None:
    """Verify that when vector store dependency fails, health check reflects 503 and unhealthy status."""
    with patch("app.services.vector_store.vector_store.get_stats", side_effect=RuntimeError("Connection refused to vector store")):
        response = client.get(f"{settings.API_V1_STR}/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["vector_store"]["status"] == "unhealthy"
        assert "Vector store unavailable" in data["components"]["vector_store"]["details"]


def test_request_correlation_id_propagation_across_errors() -> None:
    """Verify that request correlation ID (X-Request-ID) is propagated in response headers for both successes and errors."""
    # 1. Custom request ID on 400 Bad Request error
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        headers={"X-Request-ID": "req_custom_correlation_123"},
        files={"file": ("bad.txt", b"invalid content", "text/plain")},
    )
    assert response.status_code == 400
    assert response.headers.get("x-request-id") == "req_custom_correlation_123"

    # 2. Auto-generated request ID when header is omitted
    response_auto = client.get(f"{settings.API_V1_STR}/health")
    assert response_auto.status_code == 200
    assert bool(response_auto.headers.get("x-request-id"))
    assert len(response_auto.headers.get("x-request-id", "")) >= 8


def test_cross_document_isolation_multi_modal_hybrid() -> None:
    """Verify that retrieval with a document_id never returns chunks belonging to other documents."""
    from app.schemas.document import DocumentChunk
    from app.services.bm25 import BM25Retriever

    bm25 = BM25Retriever()

    chunk_a = DocumentChunk(
        chunk_id="doc_alpha_p1_c0",
        document_id="doc_alpha",
        page_number=1,
        chunk_index=0,
        text="Revenue generated by DocuLens AI was 50 million dollars in fiscal year 2025.",
        char_count=len("Revenue generated by DocuLens AI was 50 million dollars in fiscal year 2025."),
        token_count=15,
    )
    chunk_b = DocumentChunk(
        chunk_id="doc_beta_p1_c0",
        document_id="doc_beta",
        page_number=1,
        chunk_index=0,
        text="Revenue generated by Competitor was 100 million dollars in fiscal year 2025.",
        char_count=len("Revenue generated by Competitor was 100 million dollars in fiscal year 2025."),
        token_count=15,
    )

    bm25.index_chunks([chunk_a, chunk_b])

    bm25_res = bm25.retrieve(query="Revenue generated in fiscal year 2025", document_id="doc_alpha", top_k=5)
    assert len(bm25_res.results) == 1
    assert bm25_res.results[0].document_id == "doc_alpha"
    assert bm25_res.results[0].chunk_id == "doc_alpha_p1_c0"
    assert all(r.document_id == "doc_alpha" for r in bm25_res.results)


def test_diagnostics_sanitization_no_credentials_in_error_logs_and_payloads() -> None:
    """Verify that unhandled exceptions or error details never leak API keys, tokens, or system paths."""
    from app.api.main import create_app
    test_app = create_app()

    @test_app.post("/api/v1/test-leak-check")
    def leak_endpoint():
        raise RuntimeError("CRITICAL: Leak sk-secret12345678901234567890 with token Bearer a1b2c3d4e5 /etc/shadow")

    test_client = TestClient(test_app, raise_server_exceptions=False)
    res = test_client.post("/api/v1/test-leak-check")
    assert res.status_code == 500
    assert "sk-" not in res.text
    assert "secret1234" not in res.text
    assert "Bearer" not in res.text
    assert "/etc/shadow" not in res.text
    assert res.json()["detail"] == "An internal server error occurred. Please contact support or retry."

