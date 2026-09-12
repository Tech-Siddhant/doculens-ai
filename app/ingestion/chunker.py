from app.schemas.documents import ChunkingResult, DocumentChunk, ExtractionResult


def chunk_document(
    extraction_result: ExtractionResult,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> ChunkingResult:
    """Splits extracted document pages into deterministic text chunks.

    Each chunk preserves document_id, page_number, chunk_id, and chunk_index.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size.")

    chunks = []
    document_id = extraction_result.document_id

    for page in extraction_result.pages:
        text = page.text.strip()
        if not text:
            continue

        page_number = page.page_number
        text_len = len(text)

        if text_len <= chunk_size:
            chunk_id = f"{document_id}_p{page_number}_c0"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=0,
                    text=text,
                    character_count=text_len,
                )
            )
            continue

        step = chunk_size - chunk_overlap
        start = 0
        chunk_index = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]

            chunk_id = f"{document_id}_p{page_number}_c{chunk_index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    character_count=len(chunk_text),
                )
            )

            if end == text_len:
                break

            start += step
            chunk_index += 1

    return ChunkingResult(
        document_id=document_id,
        total_chunks=len(chunks),
        chunks=chunks,
    )
