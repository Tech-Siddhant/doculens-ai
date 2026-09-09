from pathlib import Path
import pytest

from app.core.config import settings
from app.schemas.embedding import VisualPageEmbedding
from app.services.visual_vector_store import (
    VisualQdrantVectorStore,
    page_id_to_point_id,
)


@pytest.fixture
def memory_visual_vector_store() -> VisualQdrantVectorStore:
    """Fixture providing a fresh in-memory visual vector store."""
    store = VisualQdrantVectorStore(location=":memory:", collection_name="test_visual_pages")
    yield store
    store.close()


def make_visual_page(
    doc_id: str,
    page: int,
    dim: int = 512,
) -> VisualPageEmbedding:
    return VisualPageEmbedding(
        document_id=doc_id,
        page_number=page,
        embedding=[0.01 * ((i + page) % 100) for i in range(dim)],
        dimension=dim,
    )


def test_deterministic_visual_point_id() -> None:
    doc_id = "doc_1234567890ab"
    id1 = page_id_to_point_id(doc_id, 1)
    id2 = page_id_to_point_id(doc_id, 1)
    assert id1 == id2
    assert isinstance(id1, str)
    assert len(id1) == 36

    diff_page_id = page_id_to_point_id(doc_id, 2)
    assert id1 != diff_page_id

    diff_doc_id = page_id_to_point_id("doc_999999999999", 1)
    assert id1 != diff_doc_id


def test_page_id_to_point_id_invalid_raises() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        page_id_to_point_id("", 1)
    with pytest.raises(ValueError, match="non-empty string"):
        page_id_to_point_id("   ", 1)
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        page_id_to_point_id("doc_1234567890ab", 0)


def test_upsert_and_count_pages(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    doc_id = "doc_111111111111"
    pages = [make_visual_page(doc_id, 1), make_visual_page(doc_id, 2)]
    count = memory_visual_vector_store.upsert_pages(pages)
    assert count == 2
    assert memory_visual_vector_store.count_pages() == 2
    assert memory_visual_vector_store.count_pages(document_id=doc_id) == 2


def test_upsert_empty_pages_returns_zero(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    assert memory_visual_vector_store.upsert_pages([]) == 0


def test_vector_dimension_mismatch_raises(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    bad_page = make_visual_page("doc_111111111111", 1, dim=256)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        memory_visual_vector_store.upsert_pages([bad_page])


def test_search_visual_pages(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    doc_id = "doc_search_test"
    page1 = make_visual_page(doc_id, 1)
    page2 = make_visual_page(doc_id, 2)
    memory_visual_vector_store.upsert_pages([page1, page2])

    results = memory_visual_vector_store.search_visual(query_vector=page1.embedding, top_k=2)
    assert len(results) == 2
    assert results[0].page_number == 1
    assert results[0].rank == 1
    assert results[0].document_id == doc_id
    assert results[0].score > 0.99


def test_search_visual_document_isolation(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    doc_a = "doc_aaaa"
    doc_b = "doc_bbbb"
    memory_visual_vector_store.upsert_pages([make_visual_page(doc_a, 1), make_visual_page(doc_a, 2)])
    memory_visual_vector_store.upsert_pages([make_visual_page(doc_b, 1)])

    query_vec = [0.01 * (i % 100) for i in range(512)]
    results = memory_visual_vector_store.search_visual(query_vector=query_vec, document_id=doc_a, top_k=5)
    assert len(results) == 2
    assert all(r.document_id == doc_a for r in results)


def test_search_visual_invalid_dimension_raises(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    with pytest.raises(ValueError, match="Query vector dimension mismatch"):
        memory_visual_vector_store.search_visual(query_vector=[0.1] * 128)


def test_search_visual_invalid_top_k_raises(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    query_vec = [0.1] * 512
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        memory_visual_vector_store.search_visual(query_vector=query_vec, top_k=0)


def test_delete_by_document(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    doc_a = "doc_del_a"
    doc_b = "doc_del_b"
    memory_visual_vector_store.upsert_pages([make_visual_page(doc_a, 1), make_visual_page(doc_a, 2)])
    memory_visual_vector_store.upsert_pages([make_visual_page(doc_b, 1)])

    assert memory_visual_vector_store.count_pages() == 3
    deleted = memory_visual_vector_store.delete_by_document(doc_a)
    assert deleted == 2
    assert memory_visual_vector_store.count_pages() == 1
    assert memory_visual_vector_store.count_pages(document_id=doc_a) == 0
    assert memory_visual_vector_store.count_pages(document_id=doc_b) == 1

    assert memory_visual_vector_store.delete_by_document("doc_none") == 0


def test_delete_by_document_empty_raises(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        memory_visual_vector_store.delete_by_document("")


def test_local_disk_persistence(tmp_path: Path) -> None:
    disk_path = str(tmp_path / "qdrant_visual_test")
    doc_id = "doc_persist"
    page = make_visual_page(doc_id, 1)

    store1 = VisualQdrantVectorStore(path=disk_path, collection_name="persist_vis")
    store1.upsert_pages([page])
    assert store1.count_pages() == 1
    store1.close()

    store2 = VisualQdrantVectorStore(path=disk_path, collection_name="persist_vis")
    assert store2.count_pages() == 1
    res = store2.search_visual(query_vector=page.embedding, top_k=1)
    assert len(res) == 1
    assert res[0].document_id == doc_id
    assert res[0].page_number == 1
    store2.close()


def test_stats_and_clear_collection(memory_visual_vector_store: VisualQdrantVectorStore) -> None:
    memory_visual_vector_store.upsert_pages([make_visual_page("doc_stats", 1)])
    stats = memory_visual_vector_store.get_stats()
    assert stats.total_points == 1
    assert stats.dimension == 512
    assert stats.status == "ready"

    memory_visual_vector_store.clear_collection()
    assert memory_visual_vector_store.count_pages() == 0
