from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.schemas.embedding import VisualPageEmbedding
from app.services.page_storage import get_page_storage_dir
from app.services.visual_embedder import visual_embedding_service
from app.services.visual_vector_store import visual_vector_store


def _create_test_image(path: Path, width: int = 100, height: int = 100, fmt: str = "PNG") -> None:
    """Create a minimal test image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color="blue")
    img.save(path, format=fmt)


def test_get_page_image_png_success(client: TestClient) -> None:
    doc_id = "doc_111122223333"
    page_dir = get_page_storage_dir(doc_id)
    img_path = page_dir / f"{doc_id}_p1.png"
    _create_test_image(img_path, width=200, height=200, fmt="PNG")

    response = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/1/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_get_page_image_jpeg_success(client: TestClient) -> None:
    doc_id = "doc_444455556666"
    page_dir = get_page_storage_dir(doc_id)
    img_path = page_dir / f"{doc_id}_p2.jpg"
    _create_test_image(img_path, width=150, height=150, fmt="JPEG")

    response = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/2/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_get_page_image_not_found(client: TestClient) -> None:
    doc_id = "doc_999999999999"
    response = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/1/image")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_page_image_invalid_document_id(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/documents/invalid_doc/pages/1/image")
    assert response.status_code == 400
    assert "invalid document identifier format" in response.json()["detail"].lower()


def test_get_page_image_path_traversal_prevention(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/documents/..%2F..%2Fetc%2Fpasswd/pages/1/image")
    # Path traversal in URL parameter gets routed either as 400 (if caught by format) or 404
    assert response.status_code in [400, 404]


def test_get_page_image_invalid_page_number(client: TestClient) -> None:
    doc_id = "doc_1234567890ab"
    response = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/0/image")
    assert response.status_code == 400
    assert "page_number must be >= 1" in response.json()["detail"]


def test_retrieve_visual_success(client: TestClient, monkeypatch) -> None:
    doc_id = "doc_121212121212"
    dim = settings.VISUAL_EMBEDDING_DIMENSION

    # Create dummy embeddings with distinct directional profiles
    emb1 = [1.0 if i % 2 == 0 else 0.0 for i in range(dim)]
    emb2 = [0.0 if i % 2 == 0 else 1.0 for i in range(dim)]
    pages = [
        VisualPageEmbedding(document_id=doc_id, page_number=1, embedding=emb1, dimension=dim),
        VisualPageEmbedding(document_id=doc_id, page_number=2, embedding=emb2, dimension=dim),
    ]
    visual_vector_store.upsert_pages(pages)

    # Mock visual query embedding matching page 1
    monkeypatch.setattr(
        visual_embedding_service,
        "embed_visual_query",
        lambda query: list(emb1),
    )

    response = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
        json={"query": "system architecture diagram", "top_k": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "system architecture diagram"
    assert data["document_id"] == doc_id
    assert data["top_k"] == 2
    assert data["total_results"] == 2
    assert len(data["results"]) == 2

    first_res = data["results"][0]
    assert first_res["rank"] == 1
    assert first_res["document_id"] == doc_id
    assert first_res["page_number"] == 1
    assert first_res["score"] > 0.99
    assert first_res["image_url"] == f"{settings.API_V1_STR}/documents/{doc_id}/pages/1/image"


def test_retrieve_visual_document_isolation(client: TestClient, monkeypatch) -> None:
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"
    dim = settings.VISUAL_EMBEDDING_DIMENSION

    emb_a = [0.1 * (i % 10) for i in range(dim)]
    emb_b = [0.2 * (i % 10) for i in range(dim)]

    visual_vector_store.upsert_pages([
        VisualPageEmbedding(document_id=doc_a, page_number=1, embedding=emb_a, dimension=dim),
        VisualPageEmbedding(document_id=doc_b, page_number=1, embedding=emb_b, dimension=dim),
    ])

    monkeypatch.setattr(
        visual_embedding_service,
        "embed_visual_query",
        lambda query: list(emb_a),
    )

    response = client.post(
        f"{settings.API_V1_STR}/documents/{doc_a}/retrieve/visual",
        json={"query": "document a architecture", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1
    assert all(r["document_id"] == doc_a for r in data["results"])


def test_retrieve_visual_nonexistent_document(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/doc_000000000000/retrieve/visual",
        json={"query": "nonexistent query"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_retrieve_visual_invalid_document_id(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/bad_id/retrieve/visual",
        json={"query": "test query"},
    )
    assert response.status_code == 400


def test_retrieve_visual_empty_and_whitespace_query(client: TestClient) -> None:
    doc_id = "doc_333344445555"
    dim = settings.VISUAL_EMBEDDING_DIMENSION
    visual_vector_store.upsert_pages([
        VisualPageEmbedding(document_id=doc_id, page_number=1, embedding=[0.1] * dim, dimension=dim),
    ])

    # Empty query (pydantic min_length=1 validation error 422)
    res1 = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
        json={"query": ""},
    )
    assert res1.status_code == 422

    # Whitespace-only query
    res2 = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
        json={"query": "   "},
    )
    assert res2.status_code == 400
    assert "cannot be empty" in res2.json()["detail"]


def test_retrieve_visual_invalid_top_k(client: TestClient) -> None:
    doc_id = "doc_555566667777"
    dim = settings.VISUAL_EMBEDDING_DIMENSION
    visual_vector_store.upsert_pages([
        VisualPageEmbedding(document_id=doc_id, page_number=1, embedding=[0.1] * dim, dimension=dim),
    ])

    # top_k = 0 (pydantic ge=1 validation error 422)
    res1 = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
        json={"query": "valid query", "top_k": 0},
    )
    assert res1.status_code == 422


def test_retrieve_visual_service_failure(client: TestClient, monkeypatch) -> None:
    doc_id = "doc_777788889999"
    dim = settings.VISUAL_EMBEDDING_DIMENSION
    visual_vector_store.upsert_pages([
        VisualPageEmbedding(document_id=doc_id, page_number=1, embedding=[0.1] * dim, dimension=dim),
    ])

    def failing_embedder(query: str):
        raise RuntimeError("Inference backend error")

    monkeypatch.setattr(visual_embedding_service, "embed_visual_query", failing_embedder)

    response = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/visual",
        json={"query": "trigger error"},
    )
    assert response.status_code == 500
    assert "visual retrieval failed" in response.json()["detail"].lower()


def test_get_page_image_on_demand_rendering(client: TestClient, monkeypatch) -> None:
    """Verify page image renders on demand if PDF exists but page image is not yet cached."""
    doc_id = "doc_aabbccddeeff"
    # Mock get_document_path to return a valid dummy path
    from app.schemas.document import RenderedPage
    dummy_img = Path(settings.RENDER_OUTPUT_DIR) / doc_id / f"{doc_id}_p1.png"
    _create_test_image(dummy_img, width=100, height=100, fmt="PNG")

    monkeypatch.setattr(
        "app.api.routes.documents.get_document_path",
        lambda did: dummy_img if did == doc_id else None,
    )
    monkeypatch.setattr(
        "app.services.renderer.render_page",
        lambda did, fpath, pnum: RenderedPage(
            document_id=did, page_number=pnum, image_path=str(dummy_img), width=100, height=100, format="png", size_bytes=100
        ),
    )

    response = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/1/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
