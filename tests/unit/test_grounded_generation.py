"""
Phase 5.5: Grounded Generation Tests
Tests the integrated flow from validated evidence to grounded answer generation.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.generation import GenerationResult
from app.schemas.retrieval import EvidenceItem, RetrievedChunk
from app.services.generator import AnswerGenerator, EvidenceInput
from app.services.llm_provider import (
    MockLLMProvider,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.fixture
def sample_evidence_items() -> list[EvidenceItem]:
    """Sample EvidenceItems from Phase 5.4 validation."""
    return [
        EvidenceItem(
            rank=1,
            score=0.92,
            initial_rank=1,
            initial_score=0.88,
            document_id="doc_phase5",
            page_number=3,
            chunk_id="doc_phase5_p3_c0",
            chunk_index=0,
            text="DocuLens AI uses evidence validation before generation to ensure data integrity.",
            retrieval_type="evidence",
            sources=["dense", "bm25"],
            raw_scores={"dense": 0.88, "bm25": 0.75},
            normalized_scores={"dense": 0.92, "bm25": 0.80},
            metadata={"char_count": 80},
        ),
        EvidenceItem(
            rank=2,
            score=0.85,
            initial_rank=3,
            initial_score=0.72,
            document_id="doc_phase5",
            page_number=5,
            chunk_id="doc_phase5_p5_c1",
            chunk_index=1,
            text="Grounded generation prevents hallucination by constraining LLM output to retrieved evidence.",
            retrieval_type="evidence",
            sources=["dense"],
            raw_scores={"dense": 0.72},
            normalized_scores={"dense": 0.85},
            metadata={"char_count": 90},
        ),
    ]


@pytest.fixture
def sample_visual_evidence() -> list[EvidenceItem]:
    """Sample visual evidence with no text."""
    return [
        EvidenceItem(
            rank=1,
            score=0.88,
            initial_rank=1,
            initial_score=0.88,
            document_id="doc_visual",
            page_number=2,
            chunk_id=None,
            chunk_index=None,
            text=None,  # Visual evidence has no text
            image_url="/api/documents/doc_visual/pages/2/image",
            retrieval_type="visual",
            sources=["visual"],
            raw_scores={"visual": 0.88},
            normalized_scores={"visual": 0.88},
            metadata={"image_width": 1200, "image_height": 800},
        ),
    ]


def test_generate_with_validated_evidence(sample_evidence_items: list[EvidenceItem]) -> None:
    """Test generation using EvidenceItem from validation layer."""
    generator = AnswerGenerator(provider=MockLLMProvider())
    
    result = generator.generate_answer(
        question="How does DocuLens ensure answer quality?",
        evidence=sample_evidence_items,
        document_id="doc_phase5",
    )
    
    assert isinstance(result, GenerationResult)
    assert result.question == "How does DocuLens ensure answer quality?"
    assert result.document_id == "doc_phase5"
    assert result.provider == "mock"
    assert len(result.citations) > 0
    assert result.is_grounded is True
    assert result.citations[0].document_id == "doc_phase5"
    assert result.citations[0].page_number == 3


def test_generate_with_retrieved_chunks() -> None:
    """Test backward compatibility with RetrievedChunk input."""
    evidence = [
        RetrievedChunk(
            rank=1,
            score=0.85,
            chunk_id="doc_test_p1_c0",
            document_id="doc_test",
            page_number=1,
            chunk_index=0,
            text="Test content for backward compatibility.",
            metadata={"char_count": 40},
        ),
    ]
    
    generator = AnswerGenerator(provider=MockLLMProvider())
    result = generator.generate_answer(
        question="What is this test about?",
        evidence=evidence,
    )
    
    assert result.is_grounded is True
    assert len(result.citations) > 0
    assert result.citations[0].chunk_id == "doc_test_p1_c0"


def test_generate_with_visual_evidence(sample_visual_evidence: list[EvidenceItem]) -> None:
    """Test generation with visual evidence (no text)."""
    generator = AnswerGenerator(provider=MockLLMProvider())
    
    result = generator.generate_answer(
        question="What does this diagram show?",
        evidence=sample_visual_evidence,
    )
    
    assert result.is_grounded is True
    assert len(result.citations) > 0
    assert result.citations[0].page_number == 2
    # Visual evidence should be handled gracefully
    assert result.citations[0].evidence_text == ""  # No text for visual


def test_generate_with_mixed_evidence() -> None:
    """Test generation with mixed text and visual evidence."""
    evidence = [
        EvidenceItem(
            rank=1,
            score=0.90,
            initial_rank=1,
            initial_score=0.90,
            document_id="doc_mixed",
            page_number=1,
            chunk_id="doc_mixed_p1_c0",
            chunk_index=0,
            text="This is textual evidence.",
            retrieval_type="evidence",
            sources=["dense"],
            raw_scores={"dense": 0.90},
            normalized_scores={"dense": 0.90},
        ),
        EvidenceItem(
            rank=2,
            score=0.85,
            initial_rank=2,
            initial_score=0.85,
            document_id="doc_mixed",
            page_number=2,
            chunk_id=None,
            chunk_index=None,
            text=None,
            image_url="/api/documents/doc_mixed/pages/2/image",
            retrieval_type="visual",
            sources=["visual"],
            raw_scores={"visual": 0.85},
            normalized_scores={"visual": 0.85},
        ),
    ]
    
    # Use custom mock response that cites both evidence items
    custom_response = "Text info [Evidence 1] is supplemented by visual info [Evidence 2]."
    generator = AnswerGenerator(provider=MockLLMProvider(mock_response=custom_response))
    result = generator.generate_answer(
        question="What information is available?",
        evidence=evidence,
    )
    
    assert result.is_grounded is True
    assert len(result.citations) == 2
    assert result.citations[0].rank == 1
    assert result.citations[1].rank == 2


def test_provider_timeout_error() -> None:
    """Test timeout error handling."""
    from unittest.mock import Mock
    
    timeout_provider = Mock(spec=MockLLMProvider)
    timeout_provider.generate.side_effect = ProviderTimeoutError(
        "LLM provider request timed out after 30.0 seconds."
    )
    
    generator = AnswerGenerator(provider=timeout_provider)
    evidence = [
        EvidenceItem(
            rank=1,
            score=0.85,
            initial_rank=1,
            initial_score=0.85,
            document_id="doc_test",
            page_number=1,
            text="Test evidence.",
            sources=["dense"],
            raw_scores={"dense": 0.85},
            normalized_scores={"dense": 0.85},
        ),
    ]
    
    with pytest.raises(ProviderTimeoutError, match="timed out after 30.0 seconds"):
        generator.generate_answer(
            question="Test question?",
            evidence=evidence,
        )


def test_provider_rate_limit_error() -> None:
    """Test rate limit error handling."""
    from unittest.mock import Mock
    
    rate_limit_provider = Mock(spec=MockLLMProvider)
    rate_limit_provider.generate.side_effect = ProviderRateLimitError(
        "LLM provider rate limit exceeded. Please retry after waiting."
    )
    
    generator = AnswerGenerator(provider=rate_limit_provider)
    evidence = [
        EvidenceItem(
            rank=1,
            score=0.85,
            initial_rank=1,
            initial_score=0.85,
            document_id="doc_test",
            page_number=1,
            text="Test evidence.",
            sources=["dense"],
            raw_scores={"dense": 0.85},
            normalized_scores={"dense": 0.85},
        ),
    ]
    
    with pytest.raises(ProviderRateLimitError, match="rate limit exceeded"):
        generator.generate_answer(
            question="Test question?",
            evidence=evidence,
        )


def test_provider_authentication_error() -> None:
    """Test authentication error handling."""
    from unittest.mock import Mock
    
    auth_provider = Mock(spec=MockLLMProvider)
    auth_provider.generate.side_effect = ProviderAuthenticationError(
        "LLM provider authentication failed. Please check your API key."
    )
    
    generator = AnswerGenerator(provider=auth_provider)
    evidence = [
        EvidenceItem(
            rank=1,
            score=0.85,
            initial_rank=1,
            initial_score=0.85,
            document_id="doc_test",
            page_number=1,
            text="Test evidence.",
            sources=["dense"],
            raw_scores={"dense": 0.85},
            normalized_scores={"dense": 0.85},
        ),
    ]
    
    with pytest.raises(ProviderAuthenticationError, match="authentication failed"):
        generator.generate_answer(
            question="Test question?",
            evidence=evidence,
        )


def test_provider_unavailable_error() -> None:
    """Test service unavailable error handling."""
    from unittest.mock import Mock
    
    unavailable_provider = Mock(spec=MockLLMProvider)
    unavailable_provider.generate.side_effect = ProviderUnavailableError(
        "LLM provider service is temporarily unavailable (HTTP 503)."
    )
    
    generator = AnswerGenerator(provider=unavailable_provider)
    evidence = [
        EvidenceItem(
            rank=1,
            score=0.85,
            initial_rank=1,
            initial_score=0.85,
            document_id="doc_test",
            page_number=1,
            text="Test evidence.",
            sources=["dense"],
            raw_scores={"dense": 0.85},
            normalized_scores={"dense": 0.85},
        ),
    ]
    
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        generator.generate_answer(
            question="Test question?",
            evidence=evidence,
        )


def test_provider_generic_error() -> None:
    """Test generic provider error handling."""
    from unittest.mock import Mock
    
    error_provider = Mock(spec=MockLLMProvider)
    error_provider.generate.side_effect = ProviderError(
        "LLM provider request failed: ConnectionError: Connection refused"
    )
    
    generator = AnswerGenerator(provider=error_provider)
    evidence = [
        EvidenceItem(
            rank=1,
            score=0.85,
            initial_rank=1,
            initial_score=0.85,
            document_id="doc_test",
            page_number=1,
            text="Test evidence.",
            sources=["dense"],
            raw_scores={"dense": 0.85},
            normalized_scores={"dense": 0.85},
        ),
    ]
    
    with pytest.raises(ProviderError, match="request failed"):
        generator.generate_answer(
            question="Test question?",
            evidence=evidence,
        )


def test_insufficient_evidence_without_calling_llm() -> None:
    """Test that empty evidence returns insufficient evidence message without calling LLM."""
    from unittest.mock import Mock
    
    mock_provider = Mock(spec=MockLLMProvider)
    generator = AnswerGenerator(provider=mock_provider)
    
    result = generator.generate_answer(
        question="Test question?",
        evidence=[],
    )
    
    assert result.answer == "I do not have sufficient information in the provided document to answer this question."
    assert result.is_grounded is False
    assert len(result.citations) == 0
    assert result.provider == "rule_based"
    mock_provider.generate.assert_not_called()  # Should NOT call LLM


def test_no_api_key_leakage_in_errors() -> None:
    """Test that provider errors do not leak API keys."""
    from app.services.llm_provider import OpenAICompatibleProvider
    
    provider = OpenAICompatibleProvider(
        api_key="sk-secret-api-key-12345",
        base_url="https://api.example.com",
        model="test-model",
    )
    
    with patch("httpx.Client.post", side_effect=Exception("Connection error occurred")):
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("Test prompt")
        
        # Verify API key is not in error message
        assert "sk-secret-api-key-12345" not in str(exc_info.value)
        assert "secret" not in str(exc_info.value).lower()


def test_http_status_code_handling() -> None:
    """Test specific HTTP error code handling."""
    from app.services.llm_provider import OpenAICompatibleProvider
    import httpx
    
    provider = OpenAICompatibleProvider(api_key="test-key", model="test-model")
    
    # Test 429 Rate Limit
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429", request=MagicMock(), response=mock_resp_429
    )
    
    with patch("httpx.Client.post", return_value=mock_resp_429):
        with pytest.raises(ProviderRateLimitError):
            provider.generate("Test prompt")
    
    # Test 401 Authentication
    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    mock_resp_401.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=mock_resp_401
    )
    
    with patch("httpx.Client.post", return_value=mock_resp_401):
        with pytest.raises(ProviderAuthenticationError):
            provider.generate("Test prompt")
    
    # Test 403 Forbidden
    mock_resp_403 = MagicMock()
    mock_resp_403.status_code = 403
    mock_resp_403.raise_for_status.side_effect = httpx.HTTPStatusError(
        "403", request=MagicMock(), response=mock_resp_403
    )
    
    with patch("httpx.Client.post", return_value=mock_resp_403):
        with pytest.raises(ProviderAuthenticationError):
            provider.generate("Test prompt")
    
    # Test 503 Service Unavailable
    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=mock_resp_503
    )
    
    with patch("httpx.Client.post", return_value=mock_resp_503):
        with pytest.raises(ProviderUnavailableError):
            provider.generate("Test prompt")
