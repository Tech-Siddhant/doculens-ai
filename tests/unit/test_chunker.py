import pytest

from app.schemas.document import DocumentMetadata, ExtractedPage, ExtractionResult
from app.services.chunker import chunk_extraction_result, chunk_text


def test_chunk_text_basic() -> None:
    text = "0123456789" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
    # chunk 0: 0..30 (len 30)
    # chunk 1: 20..50 (len 30)
    # chunk 2: 40..70 (len 30)
    # chunk 3: 60..90 (len 30)
    # chunk 4: 80..100 (len 20)
    assert len(chunks) == 5
    assert chunks[0] == text[0:30]
    assert chunks[1] == text[20:50]
    assert chunks[2] == text[40:70]
    assert chunks[3] == text[60:90]
    assert chunks[4] == text[80:100]


def test_chunk_text_short() -> None:
    text = "Short text"
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "Short text"


def test_chunk_text_empty() -> None:
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []


def test_chunk_text_exact_boundary() -> None:
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 50
    assert chunks[1] == "a" * 50


def test_chunk_text_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("hello", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("hello", chunk_size=-10, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_text("hello", chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        chunk_text("hello", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        chunk_text("hello", chunk_size=100, chunk_overlap=150)


def test_chunk_extraction_result_preserves_traceability() -> None:
    extraction = ExtractionResult(
        document_id="doc_abcdef012345",
        metadata=DocumentMetadata(
            title="Test Doc",
            author="Author",
            creation_date=None,
            total_pages=2,
            file_size_bytes=1024,
        ),
        pages=[
            ExtractedPage(page_number=1, text="Page 1 sentence one. " * 5, char_count=110),
            ExtractedPage(page_number=2, text="Page 2 content details. " * 5, char_count=125),
        ],
    )

    result = chunk_extraction_result(extraction, chunk_size=50, chunk_overlap=10)

    assert result.document_id == "doc_abcdef012345"
    assert result.chunk_size == 50
    assert result.chunk_overlap == 10
    assert result.total_chunks == len(result.chunks)
    assert len(result.chunks) > 0

    # Verify deterministic chunk IDs and traceability
    for chunk in result.chunks:
        assert chunk.document_id == "doc_abcdef012345"
        assert chunk.chunk_id == f"doc_abcdef012345_p{chunk.page_number}_c{chunk.chunk_index}"
        assert chunk.char_count == len(chunk.text)
        assert chunk.page_number in (1, 2)


def test_chunk_extraction_result_ignores_empty_pages() -> None:
    extraction = ExtractionResult(
        document_id="doc_112233445566",
        metadata=DocumentMetadata(
            title="Doc With Empty Page",
            author=None,
            creation_date=None,
            total_pages=3,
            file_size_bytes=500,
        ),
        pages=[
            ExtractedPage(page_number=1, text="Only page 1 has text", char_count=20),
            ExtractedPage(page_number=2, text="   ", char_count=3),
            ExtractedPage(page_number=3, text="", char_count=0),
        ],
    )

    result = chunk_extraction_result(extraction, chunk_size=500, chunk_overlap=50)

    assert result.total_chunks == 1
    assert result.chunks[0].chunk_id == "doc_112233445566_p1_c0"
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].text == "Only page 1 has text"
