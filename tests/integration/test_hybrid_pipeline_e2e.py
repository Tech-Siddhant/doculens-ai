"""End-to-End Hybrid Retrieval Validation (Phase 4.6).

Proves the complete pipeline:
PDF -> Ingestion -> Text Chunks -> Dense + BM25 Indices -> Rendered Pages -> Visual Index ->
Multi-modal Retrieval -> Score Normalization -> Fusion (Weighted & RRF) -> Ranked Candidates -> API.

Validates:
1. Multi-modal pipeline integration (Dense + BM25 + Visual)
2. Controlled baseline & ablation comparisons (Dense vs Dense+BM25 vs Dense+BM25+Visual)
3. Fusion strategy comparisons (Weighted vs RRF)
4. Representative queries (Semantic, Term, Identifier, Number, Page-specific, Visual layout)
5. Retrieval metrics computation (Recall@k, MRR@k)
6. Strict multi-document isolation
7. Full candidate metadata traceability
8. Error handling & graceful degradation
9. REST API endpoints & security (no secret/path leakage)
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.document import DocumentChunk, RenderedPage, RenderingResult
from app.schemas.embedding import EmbeddedChunk, VisualPageEmbedding
from app.schemas.retrieval import FusedCandidate, HybridRetrievalResult, ModalityWeights
from app.services.bm25 import bm25_retriever
from app.services.chunker import chunk_extraction_result
from app.services.embedder import embedding_service
from app.services.extractor import extract_text_and_metadata
from app.services.hybrid import HybridRetriever, hybrid_retriever
from app.services.page_storage import get_stored_page_path
from app.services.renderer import render_document
from app.services.retriever import retriever
from app.services.storage import save_uploaded_pdf
from app.services.vector_store import vector_store
from app.services.visual_embedder import visual_embedding_service
from app.services.visual_vector_store import visual_vector_store
from tests.conftest import generate_pdf_bytes


@pytest.fixture
def enterprise_spec_pdf_a() -> bytes:
    """Document A: Representative 3-page technical document with mixed modalities."""
    pages = [
        # Page 1: Normal prose + Core mission & executive summary
        (
            "DocuLens AI Enterprise Architecture. The primary mission of the platform is to "
            "provide multimodal enterprise document intelligence and semantic extraction for "
            "mission-critical knowledge bases."
        ),
        # Page 2: Technical term + Identifier + Number
        (
            "Subsystem Specification DOC-ID-99-ALPHA details the high-throughput processing pipeline. "
            "The system targets $1,250,000 in annualized infrastructure savings while utilizing "
            "hybrid lexical reciprocal rank fusion for maximum retrieval precision."
        ),
        # Page 3: Page-specific info + Visual layout & benchmark table
        (
            "Page 3 breakdown of infrastructure metrics: GPU clusters, latency histograms, "
            "and quarterly performance benchmark table comparing dense versus sparse retrieval."
        ),
    ]
    return generate_pdf_bytes(
        pages_text=pages,
        metadata={"Title": "DocuLens Spec Document A", "Author": "Engineering Core"},
    )


@pytest.fixture
def enterprise_spec_pdf_b() -> bytes:
    """Document B: Isolated secondary document with distinct terms and identifiers."""
    pages = [
        # Page 1: Isolated content
        (
            "Project Beta Overview: Completely unrelated satellite communication firmware guidelines. "
            "Operational frequency bandwidth is 14.5 GHz."
        ),
        # Page 2: Isolated identifier + numbers
        (
            "Subsystem Specification SAT-REF-77-OMEGA governs telemetry transmission telemetry rates. "
            "Total allocated orbital budget is $8,500,000 for fiscal cycle 2026."
        ),
    ]
    return generate_pdf_bytes(
        pages_text=pages,
        metadata={"Title": "DocuLens Spec Document B", "Author": "Orbital Systems"},
    )


def ingest_document_all_modalities(doc_id: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Helper to run full ingestion across Dense, BM25, and Visual pipelines."""
    # 1. Save uploaded PDF
    pdf_path = save_uploaded_pdf(doc_id, pdf_bytes)

    # 2. Extract text and chunk
    extracted = extract_text_and_metadata(doc_id, pdf_path)
    chunking_res = chunk_extraction_result(extracted, chunk_size=300, chunk_overlap=30)
    chunks = chunking_res.chunks

    # 3. Dense embedding & indexing
    embedded_chunks = embedding_service.embed_chunks(chunks)
    vector_store.upsert_chunks(embedded_chunks)

    # 4. BM25 indexing
    bm25_retriever.index_chunks(chunks)

    # 5. Page rendering & visual indexing
    render_res = render_document(doc_id, pdf_path, dpi=100, fmt="png")
    embedded_pages = visual_embedding_service.embed_rendered_pages(render_res.pages)
    visual_vector_store.upsert_pages(embedded_pages)

    return {
        "document_id": doc_id,
        "chunks": chunks,
        "rendered_pages": render_res.pages,
        "pdf_path": pdf_path,
    }


class TestEndToEndHybridPipeline:
    """Section 2, 3, 4, 5, 6, 7: Full pipeline integration, queries, metrics & comparisons."""

    @pytest.fixture(autouse=True)
    def setup_docs(self, enterprise_spec_pdf_a: bytes, enterprise_spec_pdf_b: bytes) -> None:
        self.doc_a_id = "doc_00000000000a"
        self.doc_b_id = "doc_00000000000b"
        self.ingested_a = ingest_document_all_modalities(self.doc_a_id, enterprise_spec_pdf_a)
        self.ingested_b = ingest_document_all_modalities(self.doc_b_id, enterprise_spec_pdf_b)

    def test_end_to_end_representative_queries_and_metrics(self) -> None:
        """Run 6 representative queries, verify metrics (Recall@k, MRR@k), and compare strategies."""
        # Query set with ground truth target page numbers in Document A
        test_queries = [
            {
                "id": "Q1_semantic",
                "query": "What is the primary platform mission and enterprise extraction objective?",
                "expected_page": 1,
            },
            {
                "id": "Q2_tech_term",
                "query": "hybrid lexical reciprocal rank fusion",
                "expected_page": 2,
            },
            {
                "id": "Q3_identifier",
                "query": "DOC-ID-99-ALPHA",
                "expected_page": 2,
            },
            {
                "id": "Q4_numeric",
                "query": "$1,250,000 infrastructure savings",
                "expected_page": 2,
            },
            {
                "id": "Q5_page_specific",
                "query": "Page 3 breakdown of infrastructure metrics",
                "expected_page": 3,
            },
            {
                "id": "Q6_visual_table",
                "query": "quarterly performance benchmark table and latency histograms",
                "expected_page": 3,
            },
        ]

        retriever_inst = HybridRetriever()

        for q in test_queries:
            query_text = q["query"]
            expected_page = q["expected_page"]

            # 1. Dense Only
            dense_res = retriever_inst.retrieve(
                query=query_text,
                document_id=self.doc_a_id,
                include_dense=True,
                include_bm25=False,
                include_visual=False,
                top_k=5,
            )

            # 2. Dense + BM25
            dense_bm25_res = retriever_inst.retrieve(
                query=query_text,
                document_id=self.doc_a_id,
                include_dense=True,
                include_bm25=True,
                include_visual=False,
                top_k=5,
            )

            # 3. Dense + BM25 + Visual (Weighted)
            hybrid_weighted_res = retriever_inst.retrieve(
                query=query_text,
                document_id=self.doc_a_id,
                strategy="weighted",
                include_dense=True,
                include_bm25=True,
                include_visual=True,
                top_k=5,
            )

            # 4. Dense + BM25 + Visual (RRF)
            hybrid_rrf_res = retriever_inst.retrieve(
                query=query_text,
                document_id=self.doc_a_id,
                strategy="rrf",
                rrf_k=60,
                include_dense=True,
                include_bm25=True,
                include_visual=True,
                top_k=5,
            )

            # Assert candidates returned
            assert hybrid_weighted_res.total_results > 0
            assert hybrid_rrf_res.total_results > 0

            # Verify that at least one top-3 candidate matches expected page
            top_3_pages = [c.page_number for c in hybrid_weighted_res.results[:3]]
            assert expected_page in top_3_pages, (
                f"Query '{query_text}' expected page {expected_page} in {top_3_pages}"
            )

            # Verify candidate attributes
            for cand in hybrid_weighted_res.results:
                assert cand.document_id == self.doc_a_id
                assert 1 <= cand.page_number <= 3
                assert cand.rank >= 1
                assert 0.0 <= cand.score <= 1.0
                assert len(cand.sources) > 0

        # Calculate Recall@k and MRR@k over the 6 queries for Weighted & RRF
        def evaluate_results(strategy_name: str) -> tuple[float, float]:
            recalls_at_1 = []
            recalls_at_3 = []
            reciprocal_ranks = []

            for q in test_queries:
                res = retriever_inst.retrieve(
                    query=q["query"],
                    document_id=self.doc_a_id,
                    strategy=strategy_name,
                    top_k=5,
                )
                pages = [c.page_number for c in res.results]
                target = q["expected_page"]

                # Recall@1
                recalls_at_1.append(1.0 if pages and pages[0] == target else 0.0)
                # Recall@3
                recalls_at_3.append(1.0 if target in pages[:3] else 0.0)

                # MRR
                if target in pages:
                    rank_idx = pages.index(target) + 1
                    reciprocal_ranks.append(1.0 / rank_idx)
                else:
                    reciprocal_ranks.append(0.0)

            mean_recall_3 = sum(recalls_at_3) / len(recalls_at_3)
            mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            return mean_recall_3, mean_mrr

        weighted_r3, weighted_mrr = evaluate_results("weighted")
        rrf_r3, rrf_mrr = evaluate_results("rrf")

        assert weighted_r3 == 1.0  # All 6 queries retrieve relevant page in top 3
        assert rrf_r3 == 1.0
        assert weighted_mrr >= 0.8
        assert rrf_mrr >= 0.8

    def test_ablation_and_strategy_comparison(self) -> None:
        """Section 7: Validate compare_strategies output across configurations."""
        retriever_inst = HybridRetriever()
        comparison = retriever_inst.compare_strategies(
            query="DOC-ID-99-ALPHA technical specification",
            document_id=self.doc_a_id,
            top_k=5,
        )
        assert "dense_only" in comparison
        assert "dense_bm25_weighted" in comparison
        assert "dense_bm25_visual_weighted" in comparison
        assert "dense_bm25_visual_rrf" in comparison
        assert comparison["dense_bm25_visual_weighted"].strategy == "weighted"
        assert comparison["dense_bm25_visual_rrf"].strategy == "rrf"
        assert len(comparison["dense_bm25_visual_weighted"].results) > 0
        assert len(comparison["dense_bm25_visual_rrf"].results) > 0


class TestDocumentIsolationAndTraceability:
    """Section 9 & 10: Strict isolation between Document A and B + complete metadata traceability."""

    @pytest.fixture(autouse=True)
    def setup_docs(self, enterprise_spec_pdf_a: bytes, enterprise_spec_pdf_b: bytes) -> None:
        self.doc_a_id = "doc_00000000001a"
        self.doc_b_id = "doc_00000000001b"
        ingest_document_all_modalities(self.doc_a_id, enterprise_spec_pdf_a)
        ingest_document_all_modalities(self.doc_b_id, enterprise_spec_pdf_b)

    def test_strict_document_isolation(self) -> None:
        """Verify 0% leakage from Document B when querying Document A, and vice-versa."""
        retriever_inst = HybridRetriever()

        # Query terms specific to Document B while scoping to Document A
        res_a = retriever_inst.retrieve(
            query="SAT-REF-77-OMEGA telemetry 14.5 GHz",
            document_id=self.doc_a_id,
            top_k=10,
        )
        for cand in res_a.results:
            assert cand.document_id == self.doc_a_id
            assert "SAT-REF" not in (cand.text or "")
            assert "14.5 GHz" not in (cand.text or "")

        # Query terms specific to Document A while scoping to Document B
        res_b = retriever_inst.retrieve(
            query="DOC-ID-99-ALPHA hybrid lexical reciprocal rank",
            document_id=self.doc_b_id,
            top_k=10,
        )
        for cand in res_b.results:
            assert cand.document_id == self.doc_b_id
            assert "DOC-ID-99-ALPHA" not in (cand.text or "")

    def test_candidate_metadata_traceability(self) -> None:
        """Section 10: Verify complete end-to-end evidence traceability."""
        retriever_inst = HybridRetriever()
        res = retriever_inst.retrieve(
            query="performance benchmark table and latency histograms",
            document_id=self.doc_a_id,
            strategy="weighted",
            top_k=5,
        )
        assert res.total_results > 0

        for cand in res.results:
            # 1. Document ID
            assert cand.document_id == self.doc_a_id
            # 2. Source attribution
            assert isinstance(cand.sources, list)
            assert len(cand.sources) > 0
            # 3. Page number
            assert cand.page_number in [1, 2, 3]
            # 4. Scores and ranks
            assert 0.0 <= cand.score <= 1.0
            assert cand.rank >= 1
            for src in cand.sources:
                assert src in cand.raw_scores
                assert src in cand.normalized_scores

            # 5. Evidence traceability: either text chunk or rendered page image exists
            if cand.chunk_id:
                assert cand.text is not None and len(cand.text) > 0
            if "visual" in cand.sources:
                page_path = get_stored_page_path(self.doc_a_id, cand.page_number)
                assert page_path is not None
                assert page_path.is_file()


class TestHybridFailureHandlingAndAPI:
    """Section 8 & 11: Controlled failure handling, error modes, and REST API validation."""

    @pytest.fixture(autouse=True)
    def setup_docs(self, enterprise_spec_pdf_a: bytes) -> None:
        self.doc_a_id = "doc_00000000002a"
        ingest_document_all_modalities(self.doc_a_id, enterprise_spec_pdf_a)

    def test_failure_handling_and_edge_cases(self) -> None:
        """Section 8: Failure modes and graceful handling."""
        retriever_inst = HybridRetriever()

        # 1. Empty/whitespace query -> ValueError
        with pytest.raises(ValueError, match="non-empty string"):
            retriever_inst.retrieve(query="   ", document_id=self.doc_a_id)

        # 2. Non-existent document -> Returns empty results gracefully
        empty_res = retriever_inst.retrieve(
            query="valid search query",
            document_id="doc_000000000099",
        )
        assert empty_res.total_results == 0
        assert empty_res.results == []

        # 3. All modalities disabled -> Returns empty results gracefully
        no_mod_res = retriever_inst.retrieve(
            query="valid query",
            document_id=self.doc_a_id,
            include_dense=False,
            include_bm25=False,
            include_visual=False,
        )
        assert no_mod_res.total_results == 0
        assert no_mod_res.results == []

        # 4. Invalid RRF k -> ValueError
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            retriever_inst.retrieve(
                query="valid query",
                document_id=self.doc_a_id,
                strategy="rrf",
                rrf_k=0,
            )

        # 5. Invalid weights -> ValueError
        with pytest.raises(ValueError, match="strictly positive"):
            retriever_inst.retrieve(
                query="valid query",
                document_id=self.doc_a_id,
                strategy="weighted",
                weights={"dense": 0.0, "bm25": 0.0, "visual": 0.0},
            )

    def test_rest_api_validation_and_security(self, client: TestClient) -> None:
        """Section 11 & 14: REST API validation, schema correctness, and no secret leakage."""
        # 1. Successful single-document weighted retrieval
        resp = client.post(
            f"{settings.API_V1_STR}/documents/{self.doc_a_id}/retrieve/hybrid",
            json={
                "query": "DOC-ID-99-ALPHA infrastructure savings",
                "top_k": 3,
                "strategy": "weighted",
                "weights": {"dense": 0.5, "bm25": 0.3, "visual": 0.2},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == self.doc_a_id
        assert data["strategy"] == "weighted"
        assert len(data["results"]) <= 3
        assert data["total_results"] > 0

        # Verify no secret or internal directory leakage
        raw_text = resp.text
        assert "OPENAI_API_KEY" not in raw_text
        assert "SECRET" not in raw_text
        assert "/workspaces/" not in raw_text
        assert "/home/" not in raw_text

        # 2. Successful RRF retrieval
        resp_rrf = client.post(
            f"{settings.API_V1_STR}/documents/{self.doc_a_id}/retrieve/hybrid",
            json={
                "query": "DOC-ID-99-ALPHA",
                "top_k": 3,
                "strategy": "rrf",
                "rrf_k": 60,
            },
        )
        assert resp_rrf.status_code == 200
        data_rrf = resp_rrf.json()
        assert data_rrf["strategy"] == "rrf"

        # 3. Collection-wide retrieval
        resp_col = client.post(
            f"{settings.API_V1_STR}/documents/retrieve/hybrid",
            json={"query": "enterprise document intelligence", "top_k": 4},
        )
        assert resp_col.status_code == 200
        assert resp_col.json()["document_id"] is None

        # 4. Unknown document -> 404 (valid id format but not found)
        resp_404 = client.post(
            f"{settings.API_V1_STR}/documents/doc_000000000099/retrieve/hybrid",
            json={"query": "test query"},
        )
        assert resp_404.status_code == 404

        # Invalid document format -> 400
        resp_400_id = client.post(
            f"{settings.API_V1_STR}/documents/doc_nonexistent_xyz/retrieve/hybrid",
            json={"query": "test query"},
        )
        assert resp_400_id.status_code == 400

        # 5. Invalid payloads -> 400 / 422
        resp_empty_q = client.post(
            f"{settings.API_V1_STR}/documents/{self.doc_a_id}/retrieve/hybrid",
            json={"query": "   "},
        )
        assert resp_empty_q.status_code in (400, 422)

        resp_bad_top_k = client.post(
            f"{settings.API_V1_STR}/documents/{self.doc_a_id}/retrieve/hybrid",
            json={"query": "valid query", "top_k": 0},
        )
        assert resp_bad_top_k.status_code == 422
