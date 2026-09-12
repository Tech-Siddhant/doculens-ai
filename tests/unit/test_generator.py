from unittest.mock import MagicMock, patch

import pytest

from app.schemas.generation import Citation, GenerationResult
from app.schemas.retrieval import RetrievedChunk
from app.services.generator import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AnswerGenerator,
    build_grounding_prompt,
    extract_citation_indices,
    is_refusal_response,
    validate_and_build_citations,
)
from app.services.llm_provider import (
    GoogleGeminiProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    get_llm_provider,
)


@pytest.fixture
def sample_evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            score=0.88,
            chunk_id="doc_123_p1_c0",
            document_id="doc_123",
            page_number=1,
            chunk_index=0,
            text="DocuLens AI uses dense embeddings for document QA.",
            metadata={"char_count": 52},
        ),
        RetrievedChunk(
            rank=2,
            score=0.75,
            chunk_id="doc_123_p2_c0",
            document_id="doc_123",
            page_number=2,
            chunk_index=0,
            text="Vector search ranks chunks with cosine similarity.",
            metadata={"char_count": 50},
        ),
    ]


def test_build_grounding_prompt_structure(sample_evidence: list[RetrievedChunk]) -> None:
    question = "How does DocuLens AI rank chunks?"
    prompt = build_grounding_prompt(question, sample_evidence)

    assert "STRICT GROUNDING RULES:" in prompt
    assert "--- EVIDENCE ---" in prompt
    assert "--- END EVIDENCE ---" in prompt
    assert "--- USER QUESTION ---" in prompt
    assert question in prompt
    assert "[Evidence 1] Document: doc_123 | Page: 1 | Chunk ID: doc_123_p1_c0" in prompt
    assert "[Evidence 2] Document: doc_123 | Page: 2 | Chunk ID: doc_123_p2_c0" in prompt
    assert "DocuLens AI uses dense embeddings for document QA." in prompt
    assert "Vector search ranks chunks with cosine similarity." in prompt


def test_extract_citation_indices() -> None:
    # Evidence prefix
    assert extract_citation_indices("According to [Evidence 1] and [Evidence 2].") == [1, 2]
    # Case insensitive
    assert extract_citation_indices("See [evidence 1] and [EVIDENCE 3].") == [1, 3]
    # Numeric bracket format
    assert extract_citation_indices("Dense search is used [1], cosine similarity is used [2].") == [1, 2]
    # Mixed and duplicates (preserves first occurrence order)
    assert extract_citation_indices("[Evidence 2] says X, [Evidence 1] says Y, [Evidence 2] again.") == [2, 1]
    # No citations
    assert extract_citation_indices("No brackets here.") == []


def test_validate_and_build_citations_single_valid(sample_evidence: list[RetrievedChunk]) -> None:
    answer = "DocuLens uses embeddings [Evidence 1]."
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    assert is_grounded is True
    assert len(citations) == 1
    assert citations[0].reference == "[Evidence 1]"
    assert citations[0].rank == 1
    assert citations[0].document_id == "doc_123"
    assert citations[0].page_number == 1
    assert citations[0].chunk_id == "doc_123_p1_c0"
    assert citations[0].evidence_text == sample_evidence[0].text
    assert citations[0].text == sample_evidence[0].text
    assert citations[0].score == 0.88


def test_validate_and_build_citations_multiple_valid(sample_evidence: list[RetrievedChunk]) -> None:
    answer = "DocuLens uses embeddings [Evidence 1] and vector search [Evidence 2]."
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    assert is_grounded is True
    assert len(citations) == 2
    assert citations[0].reference == "[Evidence 1]"
    assert citations[1].reference == "[Evidence 2]"
    assert citations[0].page_number == 1
    assert citations[1].page_number == 2


def test_validate_and_build_citations_fabricated_index_rejected(
    sample_evidence: list[RetrievedChunk],
) -> None:
    # Only 2 chunks exist; Evidence 99 is fabricated
    answer = "DocuLens uses quantum computers [Evidence 99]."
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    # 99 is rejected, so no valid citations remain
    assert citations == []
    assert is_grounded is False


def test_validate_and_build_citations_mixed_valid_and_fabricated(
    sample_evidence: list[RetrievedChunk],
) -> None:
    answer = "DocuLens uses embeddings [Evidence 1] and magic [Evidence 99]."
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    # Evidence 1 is preserved, Evidence 99 is discarded
    assert is_grounded is True
    assert len(citations) == 1
    assert citations[0].reference == "[Evidence 1]"
    assert citations[0].chunk_id == "doc_123_p1_c0"


def test_validate_and_build_citations_numeric_format(sample_evidence: list[RetrievedChunk]) -> None:
    answer = "DocuLens uses embeddings [1] and similarity [2]."
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    assert is_grounded is True
    assert len(citations) == 2
    assert citations[0].reference == "[Evidence 1]"
    assert citations[1].reference == "[Evidence 2]"


def test_validate_and_build_citations_refusal_response(sample_evidence: list[RetrievedChunk]) -> None:
    answer = INSUFFICIENT_EVIDENCE_ANSWER
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    assert citations == []
    assert is_grounded is False


def test_validate_and_build_citations_no_citation_tags_maps_all_chunks(
    sample_evidence: list[RetrievedChunk],
) -> None:
    answer = "DocuLens AI processes complex documents with high accuracy."
    citations, is_grounded = validate_and_build_citations(answer, sample_evidence)

    assert is_grounded is True
    assert len(citations) == 2
    assert citations[0].reference == "[Evidence 1]"
    assert citations[1].reference == "[Evidence 2]"


def test_is_refusal_response() -> None:
    assert is_refusal_response(INSUFFICIENT_EVIDENCE_ANSWER) is True
    assert is_refusal_response("There is insufficient information in the document.") is True
    assert is_refusal_response("I do not have sufficient information to answer.") is True
    assert is_refusal_response("DocuLens AI is a multimodal system [Evidence 1].") is False


def test_generate_answer_empty_evidence() -> None:
    generator = AnswerGenerator(provider=MockLLMProvider())
    result = generator.generate_answer(
        question="What is the capital of France?",
        evidence=[],
        document_id="doc_123",
    )

    assert isinstance(result, GenerationResult)
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.is_grounded is False
    assert result.citations == []
    assert result.model == "none"
    assert result.provider == "rule_based"


def test_generate_answer_successful_mock(sample_evidence: list[RetrievedChunk]) -> None:
    generator = AnswerGenerator(provider=MockLLMProvider(model="test-mock-model"))
    result = generator.generate_answer(
        question="How does retrieval work?",
        evidence=sample_evidence,
        document_id="doc_123",
    )

    assert isinstance(result, GenerationResult)
    assert result.is_grounded is True
    assert result.document_id == "doc_123"
    assert result.model == "test-mock-model"
    assert result.provider == "mock"
    assert len(result.citations) == 1
    assert result.citations[0].reference == "[Evidence 1]"
    assert result.citations[0].page_number == 1
    assert result.citations[0].chunk_id == "doc_123_p1_c0"
    assert "[Evidence 1]" in result.answer


def test_generate_answer_custom_mock_response_multiple_citations(
    sample_evidence: list[RetrievedChunk],
) -> None:
    custom_text = "DocuLens indexes documents [Evidence 1] and calculates similarity scores [Evidence 2]."
    generator = AnswerGenerator(
        provider=MockLLMProvider(mock_response=custom_text, model="custom-mock")
    )
    result = generator.generate_answer(
        question="Explain architecture.",
        evidence=sample_evidence,
    )

    assert result.answer == custom_text
    assert result.model == "custom-mock"
    assert result.is_grounded is True
    assert len(result.citations) == 2
    assert result.citations[0].reference == "[Evidence 1]"
    assert result.citations[1].reference == "[Evidence 2]"
    # Lineage and metadata preservation
    assert result.citations[0].document_id == "doc_123"
    assert result.citations[0].metadata["char_count"] == 52
    assert result.citations[1].metadata["char_count"] == 50


def test_generate_answer_provider_failure_raises(sample_evidence: list[RetrievedChunk]) -> None:
    failing_provider = MockLLMProvider(fail=True, failure_message="Upstream API timeout")
    generator = AnswerGenerator(provider=failing_provider)

    with pytest.raises(RuntimeError, match="Upstream API timeout"):
        generator.generate_answer(
            question="What is this document?",
            evidence=sample_evidence,
        )


def test_generate_answer_invalid_inputs(sample_evidence: list[RetrievedChunk]) -> None:
    generator = AnswerGenerator(provider=MockLLMProvider())

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        generator.generate_answer(question="", evidence=sample_evidence)

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        generator.generate_answer(question="   ", evidence=sample_evidence)

    with pytest.raises(ValueError, match="evidence must be a list"):
        generator.generate_answer(question="Valid question", evidence=None)  # type: ignore


def test_openai_compatible_provider_mocked_http() -> None:
    provider = OpenAICompatibleProvider(
        api_key="test-secret-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.7-flash",
        provider_name="gemini",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "gemini-3.7-flash",
        "choices": [{"message": {"content": "Grounded answer citing [Evidence 1]."}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        resp = provider.generate("Test prompt", system_prompt="System prompt")

        assert resp.content == "Grounded answer citing [Evidence 1]."
        assert resp.model == "gemini-3.7-flash"
        assert resp.provider == "gemini"
        assert resp.usage["total_tokens"] == 62
        mock_post.assert_called_once()


def test_google_gemini_provider_mocked_http() -> None:
    provider = GoogleGeminiProvider(
        api_key="test-gemini-key",
        model="gemini-3.7-flash",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "gemini-3.7-flash",
        "choices": [{"message": {"content": "Grounded answer citing [Evidence 1]."}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        resp = provider.generate("Gemini prompt")

        assert resp.content == "Grounded answer citing [Evidence 1]."
        assert resp.model == "gemini-3.7-flash"
        assert resp.provider == "gemini"
        mock_post.assert_called_once()


def test_openai_compatible_provider_http_error_does_not_leak_key() -> None:
    provider = OpenAICompatibleProvider(
        api_key="sensitive-secret-token",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.7-flash",
    )

    with patch("httpx.Client.post", side_effect=Exception("Connection refused")):
        with pytest.raises(Exception) as exc_info:
            provider.generate("Test prompt")

        assert "Connection refused" in str(exc_info.value)
        assert "sensitive-secret-token" not in str(exc_info.value)


def test_get_llm_provider_factory() -> None:
    from app.core.config import Settings

    # Default settings returns Mock provider
    mock_provider = get_llm_provider()
    assert isinstance(mock_provider, MockLLMProvider)

    # Gemini settings returns OpenAICompatibleProvider configured for Gemini
    gemini_settings = Settings(
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="test-gemini-key",
        LLM_MODEL="gemini-3.7-flash",
        _env_file=None,
    )
    gemini_provider = get_llm_provider(gemini_settings)
    assert isinstance(gemini_provider, OpenAICompatibleProvider)
    assert gemini_provider.provider_name == "gemini"
    assert gemini_provider.api_key == "test-gemini-key"
    assert gemini_provider.model == "gemini-3.7-flash"

    # OpenAI settings returns OpenAICompatibleProvider configured for OpenAI
    openai_settings = Settings(
        LLM_PROVIDER="openai",
        LLM_API_KEY="test-openai-key",
        LLM_MODEL="gpt-4o-mini",
        _env_file=None,
    )
    openai_provider = get_llm_provider(openai_settings)
    assert isinstance(openai_provider, OpenAICompatibleProvider)
    assert openai_provider.provider_name == "openai"
    assert openai_provider.api_key == "test-openai-key"
