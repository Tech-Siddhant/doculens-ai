from app.core.config import settings
from app.schemas.retrieval import RetrievalResult
from app.services.embedder import EmbeddingService, embedding_service
from app.services.vector_store import QdrantVectorStore, vector_store


class DenseRetriever:
    """Service for semantic dense vector retrieval over document chunks."""

    def __init__(
        self,
        embedder: EmbeddingService | None = None,
        store: QdrantVectorStore | None = None,
    ) -> None:
        self.embedder = embedder or embedding_service
        self.vector_store = store or vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Embed the search query and retrieve the top-k most similar document chunks.

        Preserves rank, cosine similarity score, document_id, page_number,
        chunk_id, chunk_index, text, and payload metadata.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        target_top_k = top_k if top_k is not None else settings.DEFAULT_RETRIEVAL_TOP_K
        if target_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        target_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.DEFAULT_SCORE_THRESHOLD
        )

        query_emb = self.embedder.embed_query(query)
        chunks = self.vector_store.search(
            query_vector=query_emb.embedding,
            top_k=target_top_k,
            document_id=document_id,
            score_threshold=target_threshold,
        )

        return RetrievalResult(
            query=query,
            document_id=document_id,
            top_k=target_top_k,
            total_results=len(chunks),
            results=chunks,
        )


retriever = DenseRetriever()
