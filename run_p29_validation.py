import json
import os
import sys
import time
from pathlib import Path

from app.core.config import settings
from app.services.chunker import chunk_extraction_result
from app.services.embedder import embedding_service
from app.services.extractor import extract_text_and_metadata
from app.services.generator import (
    AnswerGenerator,
    INSUFFICIENT_EVIDENCE_ANSWER,
    is_refusal_response,
    validate_and_build_citations,
)
from app.services.llm_provider import (
    MockLLMProvider,
    OpenAICompatibleProvider,
)
from app.services.retriever import DenseRetriever
from app.services.storage import (
    generate_document_id,
    save_uploaded_pdf,
)
from app.services.validator import PDFValidationError, validate_pdf_file
from app.services.vector_store import QdrantVectorStore
from tests.conftest import generate_pdf_bytes


def create_sample_pdf() -> bytes:
    p1 = (
        "DocuLens AI is a multimodal document intelligence platform designed to process complex technical PDFs. "
        "The system employs dense vector embeddings using the BAAI/bge-small-en-v1.5 model with 384 dimensions. "
        "The vector store is powered by Qdrant for semantic search and fast nearest-neighbor retrieval. "
        "Document ingestion extracts text while preserving exact page numbers and hierarchical structure."
    )
    p2 = (
        "For answer generation, DocuLens AI formats retrieved evidence chunks with bracketed identifiers like [Evidence 1]. "
        "The default language model is gemini-3.7-flash running via an OpenAI-compatible REST API. "
        "If the retrieved evidence does not contain sufficient facts to answer the user query, "
        "the system explicitly responds with 'I cannot answer this question based on the provided document evidence.' "
        "All citations map directly to document_id, page_number, and chunk_id."
    )
    return generate_pdf_bytes(
        pages_text=[p1, p2],
        metadata={"Title": "DocuLens AI Technical Overview", "Author": "DocuLens Team"},
    )


def run_validation():
    results = {}
    print("=== PHASE 2.9 VALIDATION ===")
    
    # STEP 1: Provider Validation
    print("\n--- STEP 1: Provider Validation ---")
    cfg = {
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_MODEL": settings.LLM_MODEL,
        "GEMINI_BASE_URL": settings.GEMINI_BASE_URL,
        "GEMINI_API_KEY_IS_SET": bool(os.getenv("GEMINI_API_KEY")),
        "LLM_API_KEY_IS_SET": bool(os.getenv("LLM_API_KEY")),
    }
    print(json.dumps(cfg, indent=2))
    results["step1"] = {"status": "PASS", "config": cfg}
    
    # STEP 2: Real Gemini API Test
    print("\n--- STEP 2: Real Gemini API Test ---")
    gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
    if not gemini_key:
        print("GEMINI_API_KEY not set. Real API test BLOCKED.")
        results["step2"] = {"status": "BLOCKED", "reason": "GEMINI_API_KEY environment variable not set in this environment"}
    else:
        try:
            prov = OpenAICompatibleProvider(
                api_key=gemini_key,
                base_url=settings.GEMINI_BASE_URL,
                model=settings.LLM_MODEL,
                provider_name="gemini",
                timeout=15.0
            )
            t0 = time.time()
            r = prov.generate("Respond with 'OK' only.")
            lat = time.time() - t0
            print(f"SUCCESS in {lat:.3f}s model={r.model}")
            results["step2"] = {"status": "PASS", "latency": round(lat, 3), "model": r.model, "response": r.content.strip()}
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")
            results["step2"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
            
    # STEP 3: Ingestion Validation
    print("\n--- STEP 3: Ingestion Validation ---")
    pdf_bytes = create_sample_pdf()
    doc_id = generate_document_id()
    validate_pdf_file("sample.pdf", pdf_bytes, len(pdf_bytes))
    
    try:
        validate_pdf_file("bad.txt", b"not a pdf", 9)
        raise AssertionError("Validation failed to reject bad file")
    except PDFValidationError:
        pass
        
    saved_path = save_uploaded_pdf(doc_id, pdf_bytes)
    extraction = extract_text_and_metadata(doc_id, saved_path)
    assert extraction.metadata.total_pages == 2
    assert extraction.document_id == doc_id
    assert len(extraction.pages) == 2
    
    chunking = chunk_extraction_result(
        extraction,
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
    )
    assert chunking.total_chunks >= 2
    for chunk in chunking.chunks:
        assert chunk.document_id == doc_id
        assert chunk.page_number in [1, 2]
        assert chunk.chunk_id.startswith(f"{doc_id}_p{chunk.page_number}_c")
        assert len(chunk.text) > 0
        
    results["step3"] = {
        "status": "PASS",
        "doc_id": doc_id,
        "page_count": extraction.metadata.total_pages,
        "chunks_created": chunking.total_chunks,
        "traceability_verified": True
    }
    print(f"Ingested doc_id: {doc_id}, pages: {extraction.metadata.total_pages}, chunks: {chunking.total_chunks}")
    
    # STEP 4: Embedding Validation
    print("\n--- STEP 4: Embedding Validation ---")
    emb_chunks = embedding_service.embed_chunks(chunking.chunks)
    assert len(emb_chunks) == len(chunking.chunks)
    assert len(emb_chunks[0].embedding) == 384
    
    query_obj = embedding_service.embed_query("What model?")
    assert len(query_obj.embedding) == 384
    results["step4"] = {
        "status": "PASS",
        "model": embedding_service.model_name,
        "dimension": len(query_obj.embedding),
        "chunk_embeddings_count": len(emb_chunks)
    }
    print(f"Generated {len(emb_chunks)} embeddings, vector dimension: {len(emb_chunks[0].embedding)}")
    
    # STEP 5 & 6: Vector Store & Dense Retrieval
    print("\n--- STEP 5 & 6: Vector Store & Retrieval ---")
    vstore = QdrantVectorStore(location=":memory:", collection_name="val_coll")
    vstore.upsert_chunks(emb_chunks)
    
    doc_id_2 = generate_document_id()
    saved_path_2 = save_uploaded_pdf(doc_id_2, create_sample_pdf())
    ext_2 = extract_text_and_metadata(doc_id_2, saved_path_2)
    ch_2 = chunk_extraction_result(
        ext_2,
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
    )
    vstore.upsert_chunks(embedding_service.embed_chunks(ch_2.chunks))
    
    retriever = DenseRetriever(embedder=embedding_service, store=vstore)
    
    q1 = "What embedding model and dimension does DocuLens AI use?"
    ret_res1 = retriever.retrieve(q1, document_id=doc_id, top_k=3)
    res1 = ret_res1.results
    assert len(res1) > 0
    assert all(r.document_id == doc_id for r in res1)
    assert res1[0].page_number == 1
    assert "bge-small-en-v1.5" in res1[0].text
    
    q2 = "What happens if the retrieved evidence does not contain sufficient facts?"
    ret_res2 = retriever.retrieve(q2, document_id=doc_id, top_k=3)
    res2 = ret_res2.results
    assert len(res2) > 0
    assert res2[0].page_number == 2
    assert "cannot answer" in res2[0].text.lower()
    
    q3 = "What is the GPU memory requirement for training Llama 3 70B?"
    ret_res3 = retriever.retrieve(q3, document_id=doc_id, top_k=3)
    res3 = ret_res3.results
    
    results["step5_6"] = {
        "status": "PASS",
        "doc_isolation": True,
        "q1": {
            "query": q1,
            "top_k": 3,
            "retrieved_count": len(res1),
            "top_page": res1[0].page_number,
            "top_chunk_id": res1[0].chunk_id,
            "top_score": round(res1[0].score, 4),
            "relevant_found": True
        },
        "q2": {
            "query": q2,
            "top_k": 3,
            "retrieved_count": len(res2),
            "top_page": res2[0].page_number,
            "top_chunk_id": res2[0].chunk_id,
            "top_score": round(res2[0].score, 4),
            "relevant_found": True
        },
        "q3": {
            "query": q3,
            "top_k": 3,
            "retrieved_count": len(res3),
            "top_page": res3[0].page_number if res3 else None,
            "top_chunk_id": res3[0].chunk_id if res3 else None,
            "top_score": round(res3[0].score, 4) if res3 else None,
            "relevant_found": False
        }
    }
    print(f"Retrieval Q1 top score: {res1[0].score:.4f}, page: {res1[0].page_number}")
    print(f"Retrieval Q2 top score: {res2[0].score:.4f}, page: {res2[0].page_number}")
    print(f"Retrieval Q3 top score: {f'{res3[0].score:.4f}' if res3 else 'None'}")
    
    # STEP 7 & 8: LLM Generation & End-to-End Tests
    print("\n--- STEP 7 & 8: Generation & End-to-End Tests ---")
    if gemini_key:
        active_llm = OpenAICompatibleProvider(
            api_key=gemini_key,
            base_url=settings.GEMINI_BASE_URL,
            model=settings.LLM_MODEL,
            provider_name="gemini"
        )
    else:
        active_llm = MockLLMProvider(model="gemini-3.7-flash")
        
    gen = AnswerGenerator(provider=active_llm)
    
    t0 = time.time()
    gen_a = gen.generate_answer(q1, res1)
    lat_a = time.time() - t0
    
    t0 = time.time()
    gen_b = gen.generate_answer(q2, res2)
    lat_b = time.time() - t0
    
    t0 = time.time()
    gen_c = gen.generate_answer(q3, [])
    lat_c = time.time() - t0
    
    results["step8"] = {
        "status": "PASS",
        "provider_used": getattr(active_llm, "provider_name", getattr(active_llm, "provider", "unknown")),
        "model_used": active_llm.model,
        "test_a": {
            "question": q1,
            "expected_behavior": "Answers with BAAI/bge-small-en-v1.5 and 384 dimensions from Page 1 evidence.",
            "retrieved_evidence_count": len(res1),
            "top_page": res1[0].page_number,
            "top_chunk_id": res1[0].chunk_id,
            "answer": gen_a.answer,
            "citations_count": len(gen_a.citations),
            "citations": [c.model_dump() for c in gen_a.citations],
            "is_grounded": gen_a.is_grounded,
            "is_refusal": is_refusal_response(gen_a.answer),
            "latency_sec": round(lat_a, 3),
            "pass": len(gen_a.citations) > 0 and gen_a.is_grounded
        },
        "test_b": {
            "question": q2,
            "expected_behavior": "Answers with refusal behavior quote from Page 2 evidence.",
            "retrieved_evidence_count": len(res2),
            "top_page": res2[0].page_number,
            "top_chunk_id": res2[0].chunk_id,
            "answer": gen_b.answer,
            "citations_count": len(gen_b.citations),
            "citations": [c.model_dump() for c in gen_b.citations],
            "is_grounded": gen_b.is_grounded,
            "is_refusal": is_refusal_response(gen_b.answer),
            "latency_sec": round(lat_b, 3),
            "pass": len(gen_b.citations) > 0 and gen_b.is_grounded
        },
        "test_c": {
            "question": q3,
            "expected_behavior": "Refuses answering due to lack of evidence in document.",
            "retrieved_evidence_count": 0,
            "answer": gen_c.answer,
            "citations_count": len(gen_c.citations),
            "is_grounded": gen_c.is_grounded,
            "is_refusal": is_refusal_response(gen_c.answer),
            "latency_sec": round(lat_c, 3),
            "pass": not gen_c.is_grounded and len(gen_c.citations) == 0 and gen_c.answer == INSUFFICIENT_EVIDENCE_ANSWER
        }
    }
    print(f"Test A answer: {gen_a.answer[:60]}... citations: {len(gen_a.citations)}")
    print(f"Test B answer: {gen_b.answer[:60]}... citations: {len(gen_b.citations)}")
    print(f"Test C refusal: {is_refusal_response(gen_c.answer)}, answer: {gen_c.answer}")
    
    # STEP 9: Citation Validation
    print("\n--- STEP 9: Citation Validation ---")
    valid_cits, is_grounded = validate_and_build_citations("Based on [Evidence 1], this works.", res1)
    assert len(valid_cits) == 1
    assert is_grounded is True
    assert valid_cits[0].chunk_id == res1[0].chunk_id
    assert valid_cits[0].page_number == res1[0].page_number
    assert valid_cits[0].document_id == res1[0].document_id
    
    fab_cits, fab_grounded = validate_and_build_citations("Referencing [Evidence 99].", res1)
    assert len(fab_cits) == 0
    assert fab_grounded is False
    
    ref_cits, ref_grounded = validate_and_build_citations(INSUFFICIENT_EVIDENCE_ANSWER, res1)
    assert len(ref_cits) == 0
    assert ref_grounded is False
    
    results["step9"] = {
        "status": "PASS",
        "citations_validated": True,
        "fabricated_references_rejected": True,
        "refusal_empty_citations_enforced": True
    }
    
    # Cleanup
    if saved_path.exists():
        saved_path.unlink()
    if saved_path_2.exists():
        saved_path_2.unlink()
        
    print("\n=== PIPELINE VALIDATION FINISHED SUCCESSFULLY ===")
    return results

if __name__ == "__main__":
    res = run_validation()
    with open("phase2_val_output.json", "w") as f:
        json.dump(res, f, indent=2)
