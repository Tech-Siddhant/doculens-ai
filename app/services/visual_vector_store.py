import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.core.config import settings
from app.schemas.embedding import VisualPageEmbedding
from app.schemas.retrieval import RetrievedVisualPage
from app.services.vector_store import QdrantVectorStore


def page_id_to_point_id(document_id: str, page_number: int) -> str:
    """Generate a deterministic UUIDv5 for a visual page point ID."""
    if not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")
    if page_number < 1:
        raise ValueError("page_number must be >= 1.")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doculens://visual/{document_id}/p{page_number}"))


class VisualQdrantVectorStore(QdrantVectorStore):
    """Vector store client managing visual page embeddings and metadata in Qdrant."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("collection_name", settings.QDRANT_VISUAL_COLLECTION_NAME)
        kwargs.setdefault("dimension", settings.VISUAL_EMBEDDING_DIMENSION)
        super().__init__(**kwargs)

    def upsert_pages(self, pages: Sequence[VisualPageEmbedding]) -> int:
        """Upsert a sequence of VisualPageEmbedding into Qdrant."""
        if not pages:
            return 0

        points: list[PointStruct] = []
        for idx, page in enumerate(pages):
            if len(page.embedding) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch at page {page.page_number} (index {idx}): "
                    f"expected {self.dimension}, got {len(page.embedding)}"
                )

            point_id = page_id_to_point_id(page.document_id, page.page_number)
            payload = {
                "document_id": page.document_id,
                "page_number": page.page_number,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=page.embedding,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def search_visual(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedVisualPage]:
        """Retrieve visually similar pages using vector search."""
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        top_k = top_k if top_k is not None else settings.DEFAULT_RETRIEVAL_TOP_K
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

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

        results: list[RetrievedVisualPage] = []
        for rank, pt in enumerate(query_res.points, start=1):
            payload = pt.payload or {}
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in {"document_id", "page_number"}
            }
            res_doc_id = payload.get("document_id", document_id or "")
            res_page_num = payload.get("page_number", 1)
            results.append(
                RetrievedVisualPage(
                    rank=rank,
                    score=float(pt.score),
                    document_id=res_doc_id,
                    page_number=res_page_num,
                    image_url=f"{settings.API_V1_STR}/documents/{res_doc_id}/pages/{res_page_num}/image",
                    metadata=metadata,
                )
            )

        return results

    def count_pages(self, document_id: str | None = None) -> int:
        """Count visual pages in the collection, optionally filtered by document_id."""
        return self.count_chunks(document_id=document_id)



visual_vector_store = VisualQdrantVectorStore()
