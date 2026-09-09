import math
from unittest.mock import MagicMock, patch
import pytest

from app.core.config import settings
from app.schemas.document import ChunkingResult, DocumentChunk
from app.schemas.embedding import EmbeddedChunk, EmbeddingResult, QueryEmbedding
from app.services.embedder import EmbeddingService


@pytest.fixture
def embedder() -> EmbeddingService:
    return EmbeddingService()


def test_embed_single_text_success(embedder: EmbeddingService) -> None:
    text = "DocuLens AI provides accurate multimodal document processing."
    embedding = embedder.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) == settings.EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in embedding)

    # Verify unit vector normalization (L2 norm ~ 1.0)
    norm = math.sqrt(sum(x * x for x in embedding))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_texts_batch_success(embedder: EmbeddingService) -> None:
    texts = [
        "First chunk text on page one.",
        "Second chunk text on page one.",
        "Third chunk text on page two.",
    ]
    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == settings.EMBEDDING_DIMENSION
        assert all(isinstance(x, float) for x in emb)


def test_embed_empty_batch_returns_empty(embedder: EmbeddingService) -> None:
    assert embedder.embed_texts([]) == []
    assert embedder.embed_chunks([]) == []


def test_embed_empty_text_raises_value_error(embedder: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_text("")

    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_text("   \n\t  ")


def test_embed_texts_with_empty_item_raises_value_error(embedder: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="Invalid text at index 1"):
        embedder.embed_texts(["Valid text", "  ", "Another valid text"])


def test_embed_query_success(embedder: EmbeddingService) -> None:
    query = "What is the net revenue for Q3 2024?"
    result = embedder.embed_query(query)

    assert isinstance(result, QueryEmbedding)
    assert result.query == query
    assert result.model_name == settings.EMBEDDING_MODEL_NAME
    assert result.dimension == settings.EMBEDDING_DIMENSION
    assert len(result.embedding) == settings.EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in result.embedding)


def test_embed_query_empty_raises_value_error(embedder: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_query("")

    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_query("    ")



def test_embed_chunks_preserves_traceability(embedder: EmbeddingService) -> None:
    doc_id = "doc_1234567890ab"
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_p1_c0",
            document_id=doc_id,
            page_number=1,
            chunk_index=0,
            text="First chunk on page 1 detailing intro.",
            char_count=39,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p1_c1",
            document_id=doc_id,
            page_number=1,
            chunk_index=1,
            text="Second chunk on page 1 detailing background.",
            char_count=44,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p2_c0",
            document_id=doc_id,
            page_number=2,
            chunk_index=0,
            text="First chunk on page 2 detailing results.",
            char_count=40,
        ),
    ]

    embedded_chunks = embedder.embed_chunks(chunks)

    assert len(embedded_chunks) == 3
    for orig, emb in zip(chunks, embedded_chunks, strict=True):
        assert isinstance(emb, EmbeddedChunk)
        assert emb.chunk_id == orig.chunk_id
        assert emb.document_id == orig.document_id
        assert emb.page_number == orig.page_number
        assert emb.chunk_index == orig.chunk_index
        assert emb.text == orig.text
        assert emb.dimension == settings.EMBEDDING_DIMENSION
        assert len(emb.embedding) == settings.EMBEDDING_DIMENSION


def test_embed_chunking_result(embedder: EmbeddingService) -> None:
    doc_id = "doc_aabbccddee11"
    chunk = DocumentChunk(
        chunk_id=f"{doc_id}_p1_c0",
        document_id=doc_id,
        page_number=1,
        chunk_index=0,
        text="Sample document chunk text for testing embedding result wrapper.",
        char_count=64,
    )
    chunking_result = ChunkingResult(
        document_id=doc_id,
        total_chunks=1,
        chunk_size=500,
        chunk_overlap=50,
        chunks=[chunk],
    )

    embedding_result = embedder.embed_chunking_result(chunking_result)

    assert isinstance(embedding_result, EmbeddingResult)
    assert embedding_result.document_id == doc_id
    assert embedding_result.model_name == settings.EMBEDDING_MODEL_NAME
    assert embedding_result.dimension == settings.EMBEDDING_DIMENSION
    assert embedding_result.total_embeddings == 1
    assert len(embedding_result.chunks) == 1
    assert embedding_result.chunks[0].chunk_id == f"{doc_id}_p1_c0"


def test_embed_empty_chunking_result(embedder: EmbeddingService) -> None:
    doc_id = "doc_aabbccddee22"
    chunking_result = ChunkingResult(
        document_id=doc_id,
        total_chunks=0,
        chunk_size=500,
        chunk_overlap=50,
        chunks=[],
    )

    embedding_result = embedder.embed_chunking_result(chunking_result)

    assert embedding_result.document_id == doc_id
    assert embedding_result.total_embeddings == 0
    assert embedding_result.chunks == []
    assert embedding_result.dimension == settings.EMBEDDING_DIMENSION


def test_model_lazy_loading() -> None:
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    assert service._model is None

    # Access model property triggers instantiation
    model_instance = service.model
    assert model_instance is not None
    assert service._model is model_instance


def test_custom_model_configuration() -> None:
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", batch_size=16)
    assert service.model_name == "BAAI/bge-small-en-v1.5"
    assert service.batch_size == 16


def test_model_load_failure_raises_runtime_error() -> None:
    service = EmbeddingService(model_name="nonexistent/fake-model-name-12345")
    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        _ = service.model


def test_embed_text_runtime_error_on_model_failure(embedder: EmbeddingService) -> None:
    mock_model = MagicMock()
    mock_model.embed.side_effect = Exception("ONNX runtime failure")

    with patch.object(embedder, "_model", mock_model):
        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            embedder.embed_text("Some valid text")


def test_embed_query_runtime_error_on_model_failure(embedder: EmbeddingService) -> None:
    mock_model = MagicMock()
    mock_model.query_embed.side_effect = Exception("Inference timeout")

    with patch.object(embedder, "_model", mock_model):
        with pytest.raises(RuntimeError, match="Query embedding generation failed"):
            embedder.embed_query("Sample query")
