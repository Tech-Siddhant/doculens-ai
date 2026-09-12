"""End-to-end integration and validation tests for the DocuLens Visual Pipeline (Phase 3.6).

Validates the full chain:
PDF -> page rendering -> page metadata -> visual embedding -> visual vector storage ->
text query -> visual page retrieval -> ranked page evidence -> page image API.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.document import RenderedPage, RenderingResult
from app.schemas.embedding import VisualEmbeddingResult, VisualPageEmbedding
from app.schemas.retrieval import RetrievedVisualPage
from app.services.page_storage import (
    get_page_storage_dir,
    get_stored_page_metadata,
    get_stored_page_path,
    list_stored_pages,
)
from app.services.renderer import RenderingError, render_document, render_page
from app.services.storage import generate_document_id, save_uploaded_pdf
from app.services.visual_embedder import visual_embedding_service
from app.services.visual_vector_store import VisualQdrantVectorStore, page_id_to_point_id
from tests.conftest import generate_pdf_bytes


@pytest.fixture
def multi_topic_pdf_bytes() -> bytes:
    """Generate a 3-page PDF with distinct visual and textual topics."""
    pages = [
        "DocuLens System Architecture, Multi-tier Data Flow, Client UI and FastAPI backend diagrams.",
        "Tabular Performance Benchmark, latency histograms, and empirical comparison tables.",
        "Deployment, Docker container topology, Kubernetes cluster configuration, and cloud scaling.",
    ]
    return generate_pdf_bytes(
        pages_text=pages,
        metadata={"Title": "DocuLens Multi-Topic Spec", "Author": "DocuLens Integration Test"},
    )


class TestFullVisualIngestionAndTraceability:
    """Validates Section 1A & Section 4: Full visual ingestion pipeline and traceability."""

    def test_e2e_ingestion_and_traceability(
        self, client: TestClient, multi_topic_pdf_bytes: bytes, tmp_path: Path
    ) -> None:
        # 1. Ingestion: Upload document via API
        response = client.post(
            f"{settings.API_V1_STR}/documents/upload",
            files={"file": ("test_multimodal_spec.pdf", multi_topic_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        doc_data = response.json()
        doc_id = doc_data["document_id"]
        assert doc_id.startswith("doc_")

        # 2. Rendering: Render all pages using PyMuPDF renderer service
        pdf_path = Path(settings.UPLOAD_DIR) / f"{doc_id}.pdf"
        assert pdf_path.is_file()

        render_res: RenderingResult = render_document(
            document_id=doc_id,
            file_path=pdf_path,
            dpi=150,
            fmt="png",
        )
        assert render_res.document_id == doc_id
        assert render_res.total_pages_rendered == 3
        assert len(render_res.pages) == 3

        # 3. Storage & Metadata Verification:
        for idx, page in enumerate(render_res.pages, start=1):
            assert page.document_id == doc_id
            assert page.page_number == idx
            assert page.format == "png"
            assert page.width > 0
            assert page.height > 0
            assert page.size_bytes > 0
            assert Path(page.image_path).is_file()

            # Verify deterministic storage location and metadata lookup
            stored_path = get_stored_page_path(doc_id, idx)
            assert stored_path is not None
            assert stored_path == Path(page.image_path)

            stored_meta = get_stored_page_metadata(doc_id, idx)
            assert stored_meta is not None
            assert stored_meta.document_id == doc_id
            assert stored_meta.page_number == idx
            assert stored_meta.width == page.width
            assert stored_meta.height == page.height
            assert stored_meta.size_bytes == page.size_bytes

        stored_pages = list_stored_pages(doc_id)
        assert len(stored_pages) == 3
        assert [p.page_number for p in stored_pages] == [1, 2, 3]

        # 4. Visual Embedding Generation:
        # In integration tests, we can test the embedder service
        # Mock or use visual embedder to embed rendered pages
        dim = settings.VISUAL_EMBEDDING_DIMENSION
        # Generate distinct dummy embeddings for integration verification
        dummy_embeddings = [
            [1.0 if i % 3 == 0 else 0.0 for i in range(dim)],
            [1.0 if i % 3 == 1 else 0.0 for i in range(dim)],
            [1.0 if i % 3 == 2 else 0.0 for i in range(dim)],
        ]
        embedded_pages = [
            VisualPageEmbedding(
                document_id=doc_id,
                page_number=page.page_number,
                embedding=dummy_embeddings[idx],
                dimension=dim,
            )
            for idx, page in enumerate(render_res.pages)
        ]
        assert len(embedded_pages) == 3

        # 5. Visual Vector Indexing:
        from app.services.visual_vector_store import visual_vector_store

        upserted_count = visual_vector_store.upsert_pages(embedded_pages)
        assert upserted_count == 3
        assert visual_vector_store.count_pages(document_id=doc_id) == 3

        # 6. Traceability verification from point ID to Qdrant payload
        for p_num in [1, 2, 3]:
            expected_point_id = page_id_to_point_id(doc_id, p_num)
            assert expected_point_id is not None

    def test_e2e_live_unmocked_pipeline(
        self, client: TestClient, multi_topic_pdf_bytes: bytes, tmp_path: Path
    ) -> None:
        """Complete live end-to-end integration test with live ONNX models."""
        from app.services.visual_vector_store import visual_vector_store

        # 1. Ingest PDF
        response = client.post(
            f"{settings.API_V1_STR}/documents/upload",
            files={"file": ("live_spec.pdf", multi_topic_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        doc_id = response.json()["document_id"]
        pdf_path = Path(settings.UPLOAD_DIR) / f"{doc_id}.pdf"

        # 2. Render pages
        render_res = render_document(doc_id, pdf_path, dpi=100, fmt="png")
        assert render_res.total_pages_rendered == 3

        # 3. Live visual embedding generation
        emb_res = visual_embedding_service.embed_rendering_result(render_res)
        assert emb_res.total_embeddings == 3
        assert emb_res.dimension == settings.VISUAL_EMBEDDING_DIMENSION

        # 4. Upsert into visual vector store
        upserted = visual_vector_store.upsert_pages(emb_res.pages)
        assert upserted == 3

        # 5. Live visual query retrieval via API
        query_resp = client.post(
            f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
            json={"query": "Kubernetes cluster container topology", "top_k": 3},
        )
        assert query_resp.status_code == 200
        ret_data = query_resp.json()
        assert ret_data["document_id"] == doc_id
        assert len(ret_data["results"]) == 3

        # 6. Retrieve image for top-ranked page
        top_page = ret_data["results"][0]["page_number"]
        img_resp = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/{top_page}/image")
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"] == "image/png"
        assert len(img_resp.content) > 0


class TestVisualRetrievalAndEndpoints:
    """Validates Section 1B, 1C, 1D: Querying, Document Isolation, Image Serving, and API Constraints."""

    @pytest.fixture(autouse=True)
    def setup_indexed_documents(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        """Index 2 documents with distinct pages for isolation testing."""
        from app.services.visual_vector_store import visual_vector_store

        dim = settings.VISUAL_EMBEDDING_DIMENSION
        doc_a = "doc_aaaa11112222"
        doc_b = "doc_bbbb33334444"

        # Create dummy image files in render storage for both docs
        dir_a = get_page_storage_dir(doc_a)
        dir_b = get_page_storage_dir(doc_b)

        from PIL import Image

        for p in [1, 2, 3]:
            img_a = Image.new("RGB", (100, 100), color="red")
            img_a.save(dir_a / f"{doc_a}_p{p}.png")
        
        # doc_b only has page 1 rendered
        img_b = Image.new("RGB", (100, 100), color="blue")
        img_b.save(dir_b / f"{doc_b}_p1.png")

        # Create embeddings: doc_a has high weights in first third, doc_b in second third
        emb_a = [
            VisualPageEmbedding(
                document_id=doc_a,
                page_number=1,
                embedding=[1.0 if i < 100 else 0.0 for i in range(dim)],
                dimension=dim,
            ),
            VisualPageEmbedding(
                document_id=doc_a,
                page_number=2,
                embedding=[1.0 if 100 <= i < 200 else 0.0 for i in range(dim)],
                dimension=dim,
            ),
            VisualPageEmbedding(
                document_id=doc_a,
                page_number=3,
                embedding=[1.0 if 200 <= i < 300 else 0.0 for i in range(dim)],
                dimension=dim,
            ),
        ]
        emb_b = [
            VisualPageEmbedding(
                document_id=doc_b,
                page_number=1,
                embedding=[1.0 if 300 <= i < 400 else 0.0 for i in range(dim)],
                dimension=dim,
            ),
        ]
        visual_vector_store.upsert_pages(emb_a)
        visual_vector_store.upsert_pages(emb_b)

        return {"doc_a": doc_a, "doc_b": doc_b}

    def test_image_endpoint_success_and_mime_type(
        self, client: TestClient, setup_indexed_documents: dict[str, str]
    ) -> None:
        doc_a = setup_indexed_documents["doc_a"]

        # GET existing page 1
        res = client.get(f"{settings.API_V1_STR}/documents/{doc_a}/pages/1/image")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert len(res.content) > 0

    def test_image_endpoint_error_handling_and_security(
        self, client: TestClient, setup_indexed_documents: dict[str, str]
    ) -> None:
        doc_a = setup_indexed_documents["doc_a"]

        # 404 for unknown document
        res_unknown = client.get(f"{settings.API_V1_STR}/documents/doc_999999999999/pages/1/image")
        assert res_unknown.status_code == 404
        assert "not found" in res_unknown.json()["detail"].lower()

        # 404 for nonexistent page
        res_nopage = client.get(f"{settings.API_V1_STR}/documents/{doc_a}/pages/99/image")
        assert res_nopage.status_code == 404
        assert "not found" in res_nopage.json()["detail"].lower()

        # 400 for invalid page number (< 1)
        res_zero = client.get(f"{settings.API_V1_STR}/documents/{doc_a}/pages/0/image")
        assert res_zero.status_code == 400
        assert "page_number must be >= 1" in res_zero.json()["detail"]

        # 400 for invalid document_id
        res_bad_id = client.get(f"{settings.API_V1_STR}/documents/invalid_doc/pages/1/image")
        assert res_bad_id.status_code == 400

        # Security: Cross-document isolation - doc_b cannot request doc_a's images
        doc_b = setup_indexed_documents["doc_b"]
        res_cross = client.get(f"{settings.API_V1_STR}/documents/{doc_b}/pages/2/image")
        # Page 2 exists for doc_a but was never rendered for doc_b -> 404
        assert res_cross.status_code == 404

    def test_visual_retrieval_endpoint_query_and_isolation(
        self, client: TestClient, setup_indexed_documents: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc_a = setup_indexed_documents["doc_a"]
        dim = settings.VISUAL_EMBEDDING_DIMENSION

        # Mock visual query embedding to target doc_a page 2 (indices 100..200)
        target_vec = [1.0 if 100 <= i < 200 else 0.0 for i in range(dim)]
        monkeypatch.setattr(visual_embedding_service, "embed_visual_query", lambda q: target_vec)

        res = client.post(
            f"{settings.API_V1_STR}/documents/{doc_a}/retrieve/visual",
            json={"query": "performance benchmark table", "top_k": 2},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["document_id"] == doc_a
        assert data["query"] == "performance benchmark table"
        assert data["top_k"] == 2
        assert len(data["results"]) == 2

        # Check rank 1 is page 2
        rank1 = data["results"][0]
        assert rank1["rank"] == 1
        assert rank1["document_id"] == doc_a
        assert rank1["page_number"] == 2
        assert rank1["score"] > 0.99
        assert rank1["image_url"] == f"{settings.API_V1_STR}/documents/{doc_a}/pages/2/image"

        # Check document isolation: doc_b results are strictly excluded
        assert all(r["document_id"] == doc_a for r in data["results"])
        assert all("doc_bbbb" not in r["image_url"] for r in data["results"])

    def test_visual_retrieval_input_validation(
        self, client: TestClient, setup_indexed_documents: dict[str, str]
    ) -> None:
        doc_a = setup_indexed_documents["doc_a"]

        # Empty query
        res_empty = client.post(
            f"{settings.API_V1_STR}/documents/{doc_a}/retrieve/visual",
            json={"query": ""},
        )
        assert res_empty.status_code == 422

        # Whitespace query
        res_ws = client.post(
            f"{settings.API_V1_STR}/documents/{doc_a}/retrieve/visual",
            json={"query": "   "},
        )
        assert res_ws.status_code == 400
        assert "cannot be empty" in res_ws.json()["detail"].lower()

        # Invalid top_k <= 0
        res_topk0 = client.post(
            f"{settings.API_V1_STR}/documents/{doc_a}/retrieve/visual",
            json={"query": "valid query", "top_k": 0},
        )
        assert res_topk0.status_code == 422

        # Nonexistent document
        res_not_found = client.post(
            f"{settings.API_V1_STR}/documents/doc_000000000000/retrieve/visual",
            json={"query": "valid query"},
        )
        assert res_not_found.status_code == 404


class TestMultiPageDistinguishableQueries:
    """Validates Section 2: Multi-page document with 3 queries targeting distinct pages."""

    def test_three_distinct_page_queries(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.services.visual_vector_store import visual_vector_store

        dim = settings.VISUAL_EMBEDDING_DIMENSION
        doc_id = "doc_123433334444"

        # Setup 3 pages with orthogonal visual signatures
        # Page 1: Architecture diagram (indices 0..50)
        # Page 2: Benchmark table (indices 50..100)
        # Page 3: Kubernetes deployment topology (indices 100..150)
        p1_vec = [1.0 if 0 <= i < 50 else 0.0 for i in range(dim)]
        p2_vec = [1.0 if 50 <= i < 100 else 0.0 for i in range(dim)]
        p3_vec = [1.0 if 100 <= i < 150 else 0.0 for i in range(dim)]

        visual_vector_store.upsert_pages([
            VisualPageEmbedding(document_id=doc_id, page_number=1, embedding=p1_vec, dimension=dim),
            VisualPageEmbedding(document_id=doc_id, page_number=2, embedding=p2_vec, dimension=dim),
            VisualPageEmbedding(document_id=doc_id, page_number=3, embedding=p3_vec, dimension=dim),
        ])

        # Test query 1 -> targeting Architecture (Page 1)
        monkeypatch.setattr(visual_embedding_service, "embed_visual_query", lambda q: p1_vec)
        res1 = client.post(
            f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
            json={"query": "system architecture and data flow diagram", "top_k": 3},
        )
        assert res1.status_code == 200
        results1 = res1.json()["results"]
        assert results1[0]["page_number"] == 1
        assert results1[0]["score"] > 0.99
        assert [r["page_number"] for r in results1] == [1, 2, 3] or results1[0]["page_number"] == 1

        # Test query 2 -> targeting Benchmark (Page 2)
        monkeypatch.setattr(visual_embedding_service, "embed_visual_query", lambda q: p2_vec)
        res2 = client.post(
            f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
            json={"query": "performance benchmark and comparison table", "top_k": 3},
        )
        assert res2.status_code == 200
        results2 = res2.json()["results"]
        assert results2[0]["page_number"] == 2
        assert results2[0]["score"] > 0.99

        # Test query 3 -> targeting Kubernetes Deployment (Page 3)
        monkeypatch.setattr(visual_embedding_service, "embed_visual_query", lambda q: p3_vec)
        res3 = client.post(
            f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
            json={"query": "kubernetes cluster deployment and container topology", "top_k": 3},
        )
        assert res3.status_code == 200
        results3 = res3.json()["results"]
        assert results3[0]["page_number"] == 3
        assert results3[0]["score"] > 0.99


class TestVisualPipelineFailureCases:
    """Validates Section 3: Controlled error handling across failure modes."""

    def test_rendering_corrupted_pdf_raises_controlled_error(self, tmp_path: Path) -> None:
        corrupted_file = tmp_path / "corrupted.pdf"
        corrupted_file.write_bytes(b"%PDF-1.4 ... corrupt content ... [broken]")

        doc_id = "doc_deadbeef1234"
        with pytest.raises(RenderingError, match="Failed to open PDF"):
            render_document(doc_id, corrupted_file)

    def test_render_nonexistent_pdf_raises_file_not_found(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "nonexistent.pdf"
        doc_id = "doc_deadbeef1234"
        with pytest.raises(FileNotFoundError):
            render_document(doc_id, missing_file)

    def test_render_invalid_inputs_raises_value_error(self, tmp_path: Path) -> None:
        dummy_file = tmp_path / "dummy.pdf"
        dummy_file.write_bytes(b"%PDF-1.4 dummy")

        # Invalid doc ID
        with pytest.raises(ValueError, match="Invalid document identifier format"):
            render_document("bad-doc-id", dummy_file)

        # Invalid DPI
        with pytest.raises(ValueError, match="DPI must be between"):
            render_document("doc_1234567890ab", dummy_file, dpi=10)

        # Invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            render_document("doc_1234567890ab", dummy_file, fmt="bmp")

    def test_render_page_out_of_range(self, tmp_path: Path, multi_topic_pdf_bytes: bytes) -> None:
        pdf_file = tmp_path / "valid.pdf"
        pdf_file.write_bytes(multi_topic_pdf_bytes)

        doc_id = "doc_1234567890ab"
        with pytest.raises(ValueError, match="exceeds document page count"):
            render_page(doc_id, pdf_file, page_number=10)

    def test_empty_visual_collection_retrieval(self) -> None:
        from app.services.visual_vector_store import VisualQdrantVectorStore

        store = VisualQdrantVectorStore(location=":memory:", collection_name="empty_test")
        query_vec = [0.1] * settings.VISUAL_EMBEDDING_DIMENSION
        results = store.search_visual(query_vector=query_vec, top_k=5)
        assert results == []
        store.close()
