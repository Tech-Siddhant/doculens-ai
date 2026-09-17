import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.schemas.embedding import EmbeddedChunk
from app.schemas.retrieval import RetrievedChunk
from app.schemas.vector_store import VectorStoreStats


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 from chunk_id for stable point IDs."""
    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id must be a non-empty string.")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doculens://chunk/{chunk_id}"))


class QdrantVectorStore:
    """Vector store client managing chunk embeddings and metadata payloads in Qdrant."""

    def __init__(
        self,
        location: str | None = None,
        path: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.distance = distance

        # Resolve connection parameters
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=path)
        elif settings.QDRANT_PATH:
            Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=settings.QDRANT_PATH)
        elif settings.QDRANT_URL:
            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY.get_secret_value()
                if settings.QDRANT_API_KEY
                else None,
            )
        else:
            target_location = location or settings.QDRANT_LOCATION or ":memory:"
            if isinstance(target_location, str):
                target_location = target_location.strip("'\"").strip()
                if not target_location:
                    target_location = ":memory:"
            self._client = QdrantClient(location=target_location)

        self.ensure_collection_exists()

    @property
    def client(self) -> QdrantClient:
        return self._client
    def ensure_collection_exists(self) -> None:
        """Create the target collection if it does not already exist."""
        if not self._client.collection_exists(collection_name=self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self.distance,
                ),
            )


    def upsert_chunks(self, chunks: Sequence[EmbeddedChunk]) -> int:
        """Upsert a sequence of EmbeddedChunks into Qdrant.

        Validates vector dimensions against collection configuration.
        Returns number of points upserted.
        """
        if not chunks:
            return 0

        points: list[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            if len(chunk.embedding) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch at chunk '{chunk.chunk_id}' (index {idx}): "
                    f"expected {self.dimension}, got {len(chunk.embedding)}"
                )

            point_id = chunk_id_to_point_id(chunk.chunk_id)
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "char_count": len(chunk.text),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def get_chunk_by_id(self, chunk_id: str) -> EmbeddedChunk | None:
        """Retrieve a single chunk by its chunk_id."""
        point_id = chunk_id_to_point_id(chunk_id)
        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        vector = (
            record.vector
            if isinstance(record.vector, list)
            else (list(record.vector) if record.vector is not None else [])
        )

        return EmbeddedChunk(
            chunk_id=payload.get("chunk_id", chunk_id),
            document_id=payload.get("document_id", ""),
            page_number=payload.get("page_number", 1),
            chunk_index=payload.get("chunk_index", 0),
            text=payload.get("text", ""),
            embedding=vector,
            dimension=len(vector),
        )

    def get_chunks_by_document(self, document_id: str) -> list[EmbeddedChunk]:
        """Retrieve all embedded chunks for a specific document_id."""
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=True,
            limit=10_000,
        )

        chunks: list[EmbeddedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            vector = (
                pt.vector
                if isinstance(pt.vector, list)
                else (list(pt.vector) if pt.vector is not None else [])
            )
            chunks.append(
                EmbeddedChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    document_id=payload.get("document_id", document_id),
                    page_number=payload.get("page_number", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    embedding=vector,
                    dimension=len(vector),
                )
            )

        chunks.sort(key=lambda c: (c.page_number, c.chunk_index))
        return chunks

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Search the vector store using cosine similarity.

        Optionally filters by document_id and minimum score_threshold.
        Returns a list of RetrievedChunk objects ordered by descending similarity score.
        """
        if not query_vector:
            raise ValueError("query_vector must be non-empty.")

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        query_res = self._client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for rank, pt in enumerate(query_res.points, start=1):
            payload = pt.payload or {}
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in {"chunk_id", "document_id", "page_number", "chunk_index", "text"}
            }
            results.append(
                RetrievedChunk(
                    rank=rank,
                    score=float(pt.score),
                    chunk_id=payload.get("chunk_id", str(pt.id)),
                    document_id=payload.get("document_id", document_id or ""),
                    page_number=payload.get("page_number", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    metadata=metadata,
                )
            )

        return results

    def delete_by_document(self, document_id: str) -> int:
        """Delete all points associated with a specific document_id."""
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        count_before = self.count_chunks(document_id=document_id)
        if count_before == 0:
            return 0

        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=delete_filter,
            wait=True,
        )
        return count_before

    def count_chunks(self, document_id: str | None = None) -> int:
        """Count points in the collection, optionally filtered by document_id."""
        count_filter = None
        if document_id:
            count_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        res = self._client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        )
        return res.count

    def get_stats(self) -> VectorStoreStats:
        """Get collection health and point statistics."""
        return VectorStoreStats(
            collection_name=self.collection_name,
            total_points=self.count_chunks(),
            dimension=self.dimension,
            status="ready",
        )

    def clear_collection(self) -> None:
        """Recreate the collection, deleting all indexed points."""
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        self.ensure_collection_exists()

    def close(self) -> None:
        """Close Qdrant client connection."""
        self._client.close()


vector_store = QdrantVectorStore()

