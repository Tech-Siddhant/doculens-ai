from pathlib import Path
import pytest

from app.core.config import settings
from app.schemas.embedding import EmbeddedChunk
from app.services.vector_store import QdrantVectorStore, chunk_id_to_point_id


@pytest.fixture
def memory_vector_store() -> QdrantVectorStore:
    """Fixture providing a fresh in-memory vector store."""
    store = QdrantVectorStore(location=":memory:", collection_name="test_chunks")
    yield store
    store.close()


def make_chunk(
    doc_id: str,
    page: int,
    idx: int,
    text: str = "Sample chunk text",
    dim: int = 384,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=f"{doc_id}_p{page}_c{idx}",
        document_id=doc_id,
        page_number=page,
        chunk_index=idx,
        text=text,
        embedding=[0.05 * (i + 1) for i in range(dim)],
        dimension=dim,
    )


def test_deterministic_point_id() -> None:
    chunk_id = "doc_1234567890ab_p1_c0"
    id1 = chunk_id_to_point_id(chunk_id)
    id2 = chunk_id_to_point_id(chunk_id)
    assert id1 == id2
    assert isinstance(id1, str)
    assert len(id1) == 36  # UUID standard length

    diff_id = chunk_id_to_point_id("doc_1234567890ab_p1_c1")
    assert id1 != diff_id


def test_chunk_id_to_point_id_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        chunk_id_to_point_id("")
    with pytest.raises(ValueError, match="non-empty string"):
        chunk_id_to_point_id("   ")


def test_upsert_and_retrieve_by_id(memory_vector_store: QdrantVectorStore) -> None:
    chunk = make_chunk("doc_111111111111", 1, 0, "DocuLens AI vector store testing")
    count = memory_vector_store.upsert_chunks([chunk])
    assert count == 1

    retrieved = memory_vector_store.get_chunk_by_id(chunk.chunk_id)
    assert retrieved is not None
    assert retrieved.chunk_id == chunk.chunk_id
    assert retrieved.document_id == chunk.document_id
    assert retrieved.page_number == 1
    assert retrieved.chunk_index == 0
    assert retrieved.text == "DocuLens AI vector store testing"
    assert len(retrieved.embedding) == 384


def test_retrieve_nonexistent_chunk_returns_none(
    memory_vector_store: QdrantVectorStore,
) -> None:
    assert memory_vector_store.get_chunk_by_id("nonexistent_chunk_id") is None


def test_update_existing_chunk(memory_vector_store: QdrantVectorStore) -> None:
    chunk_v1 = make_chunk("doc_111111111111", 1, 0, "Original text")
    memory_vector_store.upsert_chunks([chunk_v1])
    assert memory_vector_store.count_chunks() == 1

    chunk_v2 = make_chunk("doc_111111111111", 1, 0, "Updated text")
    memory_vector_store.upsert_chunks([chunk_v2])

    assert memory_vector_store.count_chunks() == 1
    retrieved = memory_vector_store.get_chunk_by_id(chunk_v1.chunk_id)


def test_get_chunks_by_document(memory_vector_store: QdrantVectorStore) -> None:
    doc_id = "doc_222222222222"
    chunks = [
        make_chunk(doc_id, 2, 0, "Page 2 chunk 0"),
        make_chunk(doc_id, 1, 1, "Page 1 chunk 1"),
        make_chunk(doc_id, 1, 0, "Page 1 chunk 0"),
    ]
    memory_vector_store.upsert_chunks(chunks)

    retrieved = memory_vector_store.get_chunks_by_document(doc_id)
    assert len(retrieved) == 3
    # Check sorted order: page 1 c0, page 1 c1, page 2 c0
    assert [c.chunk_id for c in retrieved] == [
        f"{doc_id}_p1_c0",
        f"{doc_id}_p1_c1",
        f"{doc_id}_p2_c0",
    ]


def test_document_isolation(memory_vector_store: QdrantVectorStore) -> None:
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"

    chunks_a = [make_chunk(doc_a, 1, 0, "Doc A chunk 0"), make_chunk(doc_a, 1, 1, "Doc A chunk 1")]
    chunks_b = [make_chunk(doc_b, 1, 0, "Doc B chunk 0"), make_chunk(doc_b, 1, 1, "Doc B chunk 1")]

    memory_vector_store.upsert_chunks(chunks_a)
    memory_vector_store.upsert_chunks(chunks_b)

    assert memory_vector_store.count_chunks() == 4
    assert memory_vector_store.count_chunks(document_id=doc_a) == 2
    assert memory_vector_store.count_chunks(document_id=doc_b) == 2

    res_a = memory_vector_store.get_chunks_by_document(doc_a)
    res_b = memory_vector_store.get_chunks_by_document(doc_b)
    assert len(res_a) == 2
    assert len(res_b) == 2
    assert all(c.document_id == doc_a for c in res_a)
    assert all(c.document_id == doc_b for c in res_b)


def test_delete_by_document(memory_vector_store: QdrantVectorStore) -> None:
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"

    memory_vector_store.upsert_chunks([make_chunk(doc_a, 1, 0), make_chunk(doc_a, 1, 1)])
    memory_vector_store.upsert_chunks([make_chunk(doc_b, 1, 0)])

    deleted_count = memory_vector_store.delete_by_document(doc_a)
    assert deleted_count == 2
    assert memory_vector_store.count_chunks(document_id=doc_a) == 0
    assert memory_vector_store.count_chunks(document_id=doc_b) == 1
    assert memory_vector_store.get_chunks_by_document(doc_a) == []

    # Deleting non-existent doc returns 0
    assert memory_vector_store.delete_by_document("doc_nonexistent") == 0


def test_upsert_empty_list_returns_zero(memory_vector_store: QdrantVectorStore) -> None:
    assert memory_vector_store.upsert_chunks([]) == 0


def test_invalid_empty_inputs_raise(memory_vector_store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        memory_vector_store.get_chunks_by_document("")
    with pytest.raises(ValueError, match="non-empty string"):
        memory_vector_store.delete_by_document("")


def test_vector_dimension_mismatch_raises_value_error(
    memory_vector_store: QdrantVectorStore,
) -> None:
    # collection expects 384, supply 128
    bad_chunk = make_chunk("doc_111111111111", 1, 0, dim=128)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        memory_vector_store.upsert_chunks([bad_chunk])


def test_local_disk_persistence(tmp_path: Path) -> None:
    disk_path = str(tmp_path / "qdrant_test_data")
    doc_id = "doc_333333333333"
    chunk = make_chunk(doc_id, 1, 0, "Persistent disk text")

    # 1. Open store on disk, upsert, and close
    store_1 = QdrantVectorStore(path=disk_path, collection_name="persistent_chunks")
    store_1.upsert_chunks([chunk])
    assert store_1.count_chunks() == 1
    store_1.close()

    # 2. Re-open store from same disk path and verify chunk persists
    store_2 = QdrantVectorStore(path=disk_path, collection_name="persistent_chunks")
    assert store_2.count_chunks() == 1
    retrieved = store_2.get_chunk_by_id(chunk.chunk_id)
    assert retrieved is not None
    assert retrieved.text == "Persistent disk text"
    store_2.close()


def test_get_stats_and_clear_collection(memory_vector_store: QdrantVectorStore) -> None:
    chunk = make_chunk("doc_444444444444", 1, 0, "Stats test")
    memory_vector_store.upsert_chunks([chunk])

    stats = memory_vector_store.get_stats()
    assert stats.total_points == 1
    assert stats.dimension == 384
    assert stats.status == "ready"

    memory_vector_store.clear_collection()
    assert memory_vector_store.count_chunks() == 0

