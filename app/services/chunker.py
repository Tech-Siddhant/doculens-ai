from app.schemas.document import ChunkingResult, DocumentChunk, ExtractionResult


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into character-based sliding-window chunks.

    Validates chunk_size > 0, chunk_overlap >= 0, chunk_overlap < chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks


def chunk_extraction_result(
    extraction_result: ExtractionResult,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> ChunkingResult:
    """Chunk extraction result page-by-page.

    Preserves complete traceability: document_id -> page_number -> chunk_id -> chunk text.
    Ignores empty pages.
    Chunk ID convention: {document_id}_p{page_number}_c{chunk_index}
    """
    all_chunks: list[DocumentChunk] = []

    for page in extraction_result.pages:
        if not page.text.strip():
            # Ignore empty pages
            continue

        raw_chunks = chunk_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, chunk_content in enumerate(raw_chunks):
            chunk_id = f"{extraction_result.document_id}_p{page.page_number}_c{chunk_idx}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=extraction_result.document_id,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    text=chunk_content,
                    char_count=len(chunk_content),
                )
            )

    return ChunkingResult(
        document_id=extraction_result.document_id,
        total_chunks=len(all_chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=all_chunks,
    )
