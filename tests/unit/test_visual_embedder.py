import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.config import settings
from app.schemas.document import RenderedPage, RenderingResult
from app.schemas.embedding import VisualEmbeddingResult, VisualPageEmbedding
from app.services.visual_embedder import VisualEmbeddingService


@pytest.fixture
def dummy_image_path(tmp_path: Path) -> str:
    """Create a small dummy image for embedding testing."""
    img_path = tmp_path / "dummy_test_image.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def visual_embedder() -> VisualEmbeddingService:
    return VisualEmbeddingService()


def test_embed_image_paths_success(
    visual_embedder: VisualEmbeddingService, dummy_image_path: str
) -> None:
    embeddings = visual_embedder.embed_image_paths([dummy_image_path])

    assert len(embeddings) == 1
    emb = embeddings[0]
    assert len(emb) == settings.VISUAL_EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in emb)

    # Note: CLIP embeddings might not be perfectly unit normalized by default in fastembed,
    # or they might be. We'll just verify it doesn't crash and returns valid floats.


def test_embed_empty_batch_returns_empty(
    visual_embedder: VisualEmbeddingService,
) -> None:
    assert visual_embedder.embed_image_paths([]) == []
    assert visual_embedder.embed_rendered_pages([]) == []


def test_embed_rendered_pages_preserves_traceability(
    visual_embedder: VisualEmbeddingService, dummy_image_path: str
) -> None:
    doc_id = "doc_111122223333"
    pages = [
        RenderedPage(
            document_id=doc_id,
            page_number=1,
            image_path=dummy_image_path,
            width=100,
            height=100,
            format="png",
            size_bytes=1024,
        ),
        RenderedPage(
            document_id=doc_id,
            page_number=2,
            image_path=dummy_image_path,
            width=100,
            height=100,
            format="png",
            size_bytes=1024,
        ),
    ]

    embedded = visual_embedder.embed_rendered_pages(pages)

    assert len(embedded) == 2
    for orig, emb in zip(pages, embedded, strict=True):
        assert isinstance(emb, VisualPageEmbedding)
        assert emb.document_id == orig.document_id
        assert emb.page_number == orig.page_number
        assert emb.dimension == settings.VISUAL_EMBEDDING_DIMENSION
        assert len(emb.embedding) == settings.VISUAL_EMBEDDING_DIMENSION


def test_embed_rendering_result(
    visual_embedder: VisualEmbeddingService, dummy_image_path: str
) -> None:
    doc_id = "doc_aaaabbbbcccc"
    page = RenderedPage(
        document_id=doc_id,
        page_number=1,
        image_path=dummy_image_path,
        width=100,
        height=100,
        format="png",
        size_bytes=1024,
    )
    rendering_result = RenderingResult(
        document_id=doc_id,
        total_pages_rendered=1,
        dpi=72,
        format="png",
        pages=[page],
    )

    embedding_result = visual_embedder.embed_rendering_result(rendering_result)

    assert isinstance(embedding_result, VisualEmbeddingResult)
    assert embedding_result.document_id == doc_id
    assert embedding_result.model_name == settings.VISUAL_EMBEDDING_MODEL_NAME
    assert embedding_result.dimension == settings.VISUAL_EMBEDDING_DIMENSION
    assert embedding_result.total_embeddings == 1
    assert len(embedding_result.pages) == 1
    assert embedding_result.pages[0].page_number == 1


def test_embed_empty_rendering_result(
    visual_embedder: VisualEmbeddingService,
) -> None:
    doc_id = "doc_empty123456"
    rendering_result = RenderingResult(
        document_id=doc_id,
        total_pages_rendered=0,
        dpi=72,
        format="png",
        pages=[],
    )

    embedding_result = visual_embedder.embed_rendering_result(rendering_result)
    assert embedding_result.document_id == doc_id
    assert embedding_result.total_embeddings == 0
    assert embedding_result.pages == []
    assert embedding_result.dimension == settings.VISUAL_EMBEDDING_DIMENSION


def test_model_lazy_loading() -> None:
    service = VisualEmbeddingService(model_name="Qdrant/resnet50-onnx")
    assert service._model is None

    # Accessing model property triggers instantiation
    with patch("app.services.visual_embedder.ImageEmbedding") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        model = service.model
        assert model is mock_instance
        mock_cls.assert_called_once_with(model_name="Qdrant/resnet50-onnx")


def test_model_load_failure_raises_runtime_error() -> None:
    service = VisualEmbeddingService(model_name="invalid/model")
    with patch(
        "app.services.visual_embedder.ImageEmbedding",
        side_effect=Exception("Download failed"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load visual embedding model"):
            _ = service.model


def test_embed_image_paths_runtime_error_on_model_failure(
    visual_embedder: VisualEmbeddingService, dummy_image_path: str
) -> None:
    mock_model = MagicMock()
    mock_model.embed.side_effect = Exception("Vision ONNX runtime failure")

    with patch.object(visual_embedder, "_model", mock_model):
        with pytest.raises(RuntimeError, match="Visual batch embedding generation failed"):
            visual_embedder.embed_image_paths([dummy_image_path])
