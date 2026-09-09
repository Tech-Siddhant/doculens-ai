from collections.abc import Sequence

from fastembed import TextEmbedding

from app.core.config import settings
from app.schemas.document import ChunkingResult, DocumentChunk
from app.schemas.embedding import EmbeddedChunk, EmbeddingResult, QueryEmbedding


class EmbeddingService:
    """Service for generating dense vector embeddings using local ONNX inference."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy-load the embedding model on first access."""
        if self._model is None:
            try:
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate dense embedding for a single non-empty text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")

        try:
            embeddings = list(self.model.embed([text], batch_size=1))
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        if not embeddings:
            raise RuntimeError("Embedding model produced no output for text.")
        return [float(x) for x in embeddings[0].tolist()]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of text strings."""
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Invalid text at index {idx}: text must be a non-empty string."
                )

        try:
            raw_embeddings = list(self.model.embed(list(texts), batch_size=self.batch_size))
        except Exception as exc:
            raise RuntimeError(f"Batch embedding generation failed: {exc}") from exc

        return [[float(x) for x in emb.tolist()] for emb in raw_embeddings]

    def embed_query(self, query: str) -> QueryEmbedding:
        """Generate dense embedding for a search query (asymmetric search prompt)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Cannot embed empty or whitespace-only query.")

        try:
            raw_embeddings = list(self.model.query_embed(query))
        except Exception as exc:
            raise RuntimeError(f"Query embedding generation failed: {exc}") from exc

        if not raw_embeddings:
            raise RuntimeError("Embedding model produced no output for query.")
        embedding = [float(x) for x in raw_embeddings[0].tolist()]

        return QueryEmbedding(
            query=query,
            model_name=self.model_name,
            dimension=len(embedding),
            embedding=embedding,
        )

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for a list of DocumentChunks while preserving traceability."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=emb,
                    dimension=len(emb),
                )
            )
        return embedded_chunks

    def embed_chunking_result(self, chunking_result: ChunkingResult) -> EmbeddingResult:
        """Generate EmbeddingResult from a complete ChunkingResult."""
        embedded_chunks = self.embed_chunks(chunking_result.chunks)
        dimension = (
            embedded_chunks[0].dimension
            if embedded_chunks
            else settings.EMBEDDING_DIMENSION
        )

        return EmbeddingResult(
            document_id=chunking_result.document_id,
            model_name=self.model_name,
            dimension=dimension,
            total_embeddings=len(embedded_chunks),
            chunks=embedded_chunks,
        )


embedding_service = EmbeddingService()
