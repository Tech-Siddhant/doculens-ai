import pytest

from app.schemas.embedding import EmbeddedChunk, QueryEmbedding
from app.schemas.retrieval import RetrievalResult
from app.services.retriever import DenseRetriever
from app.services.vector_store import QdrantVectorStore


class MockEmbeddingService:
    """Deterministic mock embedding service for unit tests."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.model_name = "mock-embedder"

    def embed_query(self, query: str) -> QueryEmbedding:
        if not query or not query.strip():
            raise ValueError("Cannot embed empty query.")

        q_lower = query.lower()
        if "machine learning" in q_lower or "ai" in q_lower:
            vector = [1.0, 0.0, 0.0, 0.0]
        elif "quantum computing" in q_lower:
            vector = [0.0, 1.0, 0.0, 0.0]
        elif "biology" in q_lower:
            vector = [0.0, 0.0, 1.0, 0.0]
        else:
            vector = [0.5, 0.5, 0.0, 0.0]

        return QueryEmbedding(
            query=query,
            model_name=self.model_name,
            dimension=self.dimension,
            embedding=vector,
        )


@pytest.fixture
def mock_embedder() -> MockEmbeddingService:
    return MockEmbeddingService(dimension=4)


@pytest.fixture
def in_memory_store() -> QdrantVectorStore:
    store = QdrantVectorStore(
        location=":memory:",
        collection_name="test_retrieval_chunks",
        dimension=4,
    )
    yield store
    store.close()


@pytest.fixture
def sample_chunks() -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk_id="doc_ml_p1_c0",
            document_id="doc_ml",
            page_number=1,
            chunk_index=0,
            text="Machine learning models learn patterns from training data.",
            embedding=[1.0, 0.0, 0.0, 0.0],
            dimension=4,
        ),
        EmbeddedChunk(
            chunk_id="doc_ml_p2_c0",
            document_id="doc_ml",
            page_number=2,
            chunk_index=0,
            text="Deep neural networks require substantial compute and data.",
            embedding=[0.8, 0.2, 0.0, 0.0],
            dimension=4,
        ),
        EmbeddedChunk(
            chunk_id="doc_quantum_p1_c0",
            document_id="doc_quantum",
            page_number=1,
            chunk_index=0,
            text="Quantum computing utilizes superposition and entanglement qubits.",
            embedding=[0.0, 1.0, 0.0, 0.0],
            dimension=4,
        ),
        EmbeddedChunk(
            chunk_id="doc_bio_p1_c0",
            document_id="doc_bio",
            page_number=1,
            chunk_index=0,
            text="Cellular biology examines mitochondrial respiration and genetics.",
            embedding=[0.0, 0.0, 1.0, 0.0],
            dimension=4,
        ),
    ]


def test_successful_retrieval(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
    sample_chunks: list[EmbeddedChunk],
) -> None:
    in_memory_store.upsert_chunks(sample_chunks)
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    result = retriever.retrieve(query="machine learning algorithms", top_k=3)

    assert isinstance(result, RetrievalResult)
    assert result.query == "machine learning algorithms"
    assert result.top_k == 3
    assert result.total_results == 3
    assert len(result.results) == 3

    top_chunk = result.results[0]
    assert top_chunk.rank == 1
    assert top_chunk.chunk_id == "doc_ml_p1_c0"
    assert top_chunk.document_id == "doc_ml"
    assert top_chunk.page_number == 1
    assert top_chunk.chunk_index == 0
    assert "Machine learning models" in top_chunk.text
    assert top_chunk.score > 0.99

    assert [c.rank for c in result.results] == [1, 2, 3]
    assert result.results[0].score >= result.results[1].score >= result.results[2].score


def test_ranking_and_scores(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
    sample_chunks: list[EmbeddedChunk],
) -> None:
    in_memory_store.upsert_chunks(sample_chunks)
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    result = retriever.retrieve(query="quantum computing circuits", top_k=2)

    assert result.total_results == 2
    assert result.results[0].chunk_id == "doc_quantum_p1_c0"
    assert result.results[0].rank == 1
    assert result.results[0].score > 0.99


def test_top_k_behavior(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
    sample_chunks: list[EmbeddedChunk],
) -> None:
    in_memory_store.upsert_chunks(sample_chunks)
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    result_k1 = retriever.retrieve(query="ai", top_k=1)
    assert result_k1.total_results == 1
    assert len(result_k1.results) == 1

    result_k4 = retriever.retrieve(query="ai", top_k=10)
    assert result_k4.total_results == 4
    assert len(result_k4.results) == 4


def test_document_id_isolation(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
    sample_chunks: list[EmbeddedChunk],
) -> None:
    in_memory_store.upsert_chunks(sample_chunks)
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    result_ml = retriever.retrieve(
        query="quantum machine learning",
        top_k=5,
        document_id="doc_ml",
    )

    assert result_ml.document_id == "doc_ml"
    assert result_ml.total_results == 2
    for chunk in result_ml.results:
        assert chunk.document_id == "doc_ml"

    result_quantum = retriever.retrieve(
        query="quantum machine learning",
        top_k=5,
        document_id="doc_quantum",
    )
    assert result_quantum.total_results == 1
    assert result_quantum.results[0].document_id == "doc_quantum"


def test_score_threshold_filtering(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
    sample_chunks: list[EmbeddedChunk],
) -> None:
    in_memory_store.upsert_chunks(sample_chunks)
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    # Query for machine learning with high threshold (0.99)
    result = retriever.retrieve(
        query="machine learning",
        top_k=5,
        score_threshold=0.99,
    )

    # Only doc_ml_p1_c0 has cosine similarity = 1.0 (>= 0.99)
    assert result.total_results == 1
    assert result.results[0].chunk_id == "doc_ml_p1_c0"
    assert result.results[0].score >= 0.99


def test_empty_store_returns_empty_results(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
) -> None:
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    result = retriever.retrieve(query="any search term", top_k=5)
    assert result.total_results == 0
    assert result.results == []


def test_metadata_preservation(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
) -> None:
    chunk = EmbeddedChunk(
        chunk_id="doc_meta_p3_c2",
        document_id="doc_meta",
        page_number=3,
        chunk_index=2,
        text="Sample paragraph with rich metadata citation.",
        embedding=[1.0, 0.0, 0.0, 0.0],
        dimension=4,
    )
    in_memory_store.upsert_chunks([chunk])
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    result = retriever.retrieve(query="ai", top_k=1)
    assert result.total_results == 1
    retrieved = result.results[0]

    assert retrieved.chunk_id == "doc_meta_p3_c2"
    assert retrieved.document_id == "doc_meta"
    assert retrieved.page_number == 3
    assert retrieved.chunk_index == 2
    assert retrieved.text == "Sample paragraph with rich metadata citation."
    assert retrieved.metadata.get("char_count") == len("Sample paragraph with rich metadata citation.")


def test_invalid_query_raises_value_error(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
) -> None:
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve(query="")

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve(query="   ")


def test_invalid_top_k_raises_value_error(
    mock_embedder: MockEmbeddingService,
    in_memory_store: QdrantVectorStore,
) -> None:
    retriever = DenseRetriever(embedder=mock_embedder, store=in_memory_store)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve(query="machine learning", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve(query="machine learning", top_k=-5)
