"""Service for generating visual embeddings using local ONNX inference."""

from collections.abc import Sequence

from fastembed import ImageEmbedding

from app.core.config import settings
from app.schemas.document import RenderedPage, RenderingResult
from app.schemas.embedding import VisualEmbeddingResult, VisualPageEmbedding


class VisualEmbeddingService:
    """Service for generating visual embeddings using local ONNX inference."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.VISUAL_EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.VISUAL_EMBEDDING_BATCH_SIZE
        self._model: ImageEmbedding | None = None

    @property
    def model(self) -> ImageEmbedding:
        """Lazy-load the image embedding model on first access."""
        if self._model is None:
            try:
                self._model = ImageEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load visual embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def embed_image_paths(self, image_paths: Sequence[str]) -> list[list[float]]:
        """Generate dense visual embeddings for a batch of image files."""
        if not image_paths:
            return []

        try:
            # fastembed.ImageEmbedding accepts an iterable of strings (paths) or PIL limit
            raw_embeddings = list(
                self.model.embed(list(image_paths), batch_size=self.batch_size)
            )
        except Exception as exc:
            raise RuntimeError(f"Visual batch embedding generation failed: {exc}") from exc

        return [[float(x) for x in emb.tolist()] for emb in raw_embeddings]

    def embed_visual_query(self, query: str) -> list[float]:
        """Embed a text query for visual search using the corresponding text model."""
        if not hasattr(self, "_text_model") or self._text_model is None:
            # Determine the corresponding text model for the vision model
            if self.model_name == "Qdrant/clip-ViT-B-32-vision":
                text_model_name = "Qdrant/clip-ViT-B-32-text"
            else:
                text_model_name = self.model_name.replace("-vision", "-text")
                
            from fastembed import TextEmbedding
            try:
                self._text_model = TextEmbedding(model_name=text_model_name)
            except Exception as exc:
                raise RuntimeError(f"Failed to load text embedding model '{text_model_name}': {exc}") from exc

        try:
            embeddings = list(self._text_model.embed([query]))
            if not embeddings:
                return []
            return [float(x) for x in embeddings[0].tolist()]
        except Exception as exc:
            raise RuntimeError(f"Visual query embedding failed: {exc}") from exc


    def embed_rendered_pages(
        self, pages: Sequence[RenderedPage]
    ) -> list[VisualPageEmbedding]:
        """Generate embeddings for a list of RenderedPages while preserving traceability."""
        if not pages:
            return []

        image_paths = [page.image_path for page in pages]
        embeddings = self.embed_image_paths(image_paths)

        embedded_pages: list[VisualPageEmbedding] = []
        for page, emb in zip(pages, embeddings, strict=True):
            embedded_pages.append(
                VisualPageEmbedding(
                    document_id=page.document_id,
                    page_number=page.page_number,
                    embedding=emb,
                    dimension=len(emb),
                )
            )
        return embedded_pages

    def embed_rendering_result(
        self, rendering_result: RenderingResult
    ) -> VisualEmbeddingResult:
        """Generate VisualEmbeddingResult from a complete RenderingResult."""
        embedded_pages = self.embed_rendered_pages(rendering_result.pages)
        dimension = (
            embedded_pages[0].dimension
            if embedded_pages
            else settings.VISUAL_EMBEDDING_DIMENSION
        )

        return VisualEmbeddingResult(
            document_id=rendering_result.document_id,
            model_name=self.model_name,
            dimension=dimension,
            total_embeddings=len(embedded_pages),
            pages=embedded_pages,
        )


visual_embedding_service = VisualEmbeddingService()
