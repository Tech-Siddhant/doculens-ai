"""Unit tests for page image storage and metadata service."""

from pathlib import Path
import pytest
from PIL import Image

from app.schemas.document import RenderedPage
from app.services.page_storage import (
    PageStorageError,
    delete_stored_pages,
    get_page_storage_dir,
    get_stored_page_metadata,
    get_stored_page_path,
    list_stored_pages,
)


@pytest.fixture
def doc_id():
    return "doc_1234567890ab"


@pytest.fixture
def storage_dir(tmp_path: Path):
    """Create a temporary storage directory."""
    return tmp_path / "rendered_pages"


def _create_test_image(path: Path, width: int = 100, height: int = 100):
    """Create a minimal test image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color="white")
    img.save(path)


def test_get_page_storage_dir_valid(doc_id: str, storage_dir: Path):
    result = get_page_storage_dir(doc_id, base_dir=str(storage_dir))
    assert result == storage_dir / doc_id
    assert result.exists()


def test_get_page_storage_dir_invalid_id(storage_dir: Path):
    with pytest.raises(ValueError, match="Invalid document identifier"):
        get_page_storage_dir("bad-id", base_dir=str(storage_dir))


def test_get_stored_page_path_exists(doc_id: str, storage_dir: Path):
    page_dir = storage_dir / doc_id
    page_dir.mkdir(parents=True)
    test_image = page_dir / f"{doc_id}_p1.png"
    _create_test_image(test_image)
    
    result = get_stored_page_path(doc_id, 1, fmt="png", base_dir=str(storage_dir))
    assert result == test_image


def test_get_stored_page_path_missing(doc_id: str, storage_dir: Path):
    result = get_stored_page_path(doc_id, 1, fmt="png", base_dir=str(storage_dir))
    assert result is None


def test_get_stored_page_path_invalid_inputs(storage_dir: Path):
    with pytest.raises(ValueError, match="Invalid document identifier"):
        get_stored_page_path("bad-id", 1, base_dir=str(storage_dir))
    
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        get_stored_page_path("doc_1234567890ab", 0, base_dir=str(storage_dir))


def test_get_stored_page_metadata_exists(doc_id: str, storage_dir: Path):
    page_dir = storage_dir / doc_id
    page_dir.mkdir(parents=True)
    test_image = page_dir / f"{doc_id}_p1.png"
    _create_test_image(test_image, width=200, height=300)
    
    result = get_stored_page_metadata(
        doc_id, 1, fmt="png", base_dir=str(storage_dir)
    )
    
    assert isinstance(result, RenderedPage)
    assert result.document_id == doc_id
    assert result.page_number == 1
    assert result.width == 200
    assert result.height == 300
    assert result.format == "png"
    assert result.size_bytes > 0


def test_get_stored_page_metadata_missing(doc_id: str, storage_dir: Path):
    result = get_stored_page_metadata(
        doc_id, 1, fmt="png", base_dir=str(storage_dir)
    )
    assert result is None


def test_list_stored_pages_empty(doc_id: str, storage_dir: Path):
    result = list_stored_pages(doc_id, base_dir=str(storage_dir))
    assert result == []


def test_list_stored_pages_multiple(doc_id: str, storage_dir: Path):
    page_dir = storage_dir / doc_id
    page_dir.mkdir(parents=True)
    
    # Create multiple pages
    _create_test_image(page_dir / f"{doc_id}_p1.png", width=100, height=100)
    _create_test_image(page_dir / f"{doc_id}_p2.png", width=200, height=200)
    _create_test_image(page_dir / f"{doc_id}_p3.jpg", width=300, height=300)
    
    result = list_stored_pages(doc_id, base_dir=str(storage_dir))
    
    assert len(result) == 3
    assert result[0].page_number == 1
    assert result[1].page_number == 2
    assert result[2].page_number == 3
    assert result[0].format == "png"
    assert result[2].format == "jpeg"


def test_list_stored_pages_invalid_id(storage_dir: Path):
    with pytest.raises(ValueError, match="Invalid document identifier"):
        list_stored_pages("bad-id", base_dir=str(storage_dir))


def test_delete_stored_pages_success(doc_id: str, storage_dir: Path):
    page_dir = storage_dir / doc_id
    page_dir.mkdir(parents=True)
    
    # Create multiple pages
    _create_test_image(page_dir / f"{doc_id}_p1.png")
    _create_test_image(page_dir / f"{doc_id}_p2.png")
    
    assert page_dir.exists()
    
    deleted_count = delete_stored_pages(doc_id, base_dir=str(storage_dir))
    
    assert deleted_count == 2
    assert not page_dir.exists()


def test_delete_stored_pages_missing(doc_id: str, storage_dir: Path):
    deleted_count = delete_stored_pages(doc_id, base_dir=str(storage_dir))
    assert deleted_count == 0


def test_delete_stored_pages_invalid_id(storage_dir: Path):
    with pytest.raises(ValueError, match="Invalid document identifier"):
        delete_stored_pages("bad-id", base_dir=str(storage_dir))


def test_page_metadata_jpeg_format(doc_id: str, storage_dir: Path):
    page_dir = storage_dir / doc_id
    page_dir.mkdir(parents=True)
    test_image = page_dir / f"{doc_id}_p1.jpg"
    _create_test_image(test_image, width=150, height=250)
    
    result = get_stored_page_metadata(
        doc_id, 1, fmt="jpeg", base_dir=str(storage_dir)
    )
    
    assert result is not None
    assert result.format == "jpeg"
    assert result.width == 150
    assert result.height == 250
