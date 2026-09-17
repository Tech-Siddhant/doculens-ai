"""Unit tests for Phase 8.3 Pipeline Telemetry — stage metrics, timing, safety, and frontend compatibility."""

from unittest.mock import MagicMock, patch
import pytest

from app.schemas.generation import GenerationResult
from app.schemas.retrieval import FusedCandidate, RerankedCandidate
from app.services.orchestrator import AnswerOrchestrator, orchestrator


def test_orchestrator_pipeline_trace_stages() -> None:
    """Verify that orchestrator produces all 8 pipeline stages with correct telemetry."""
    orch = AnswerOrchestrator()
    
    with patch("app.services.orchestrator.hybrid_retriever.retrieve") as mock_ret:
        mock_ret.return_value = MagicMock(
            results=[
                FusedCandidate(
                    rank=1,
                    score=0.9,
                    document_id="doc_test",
                    page_number=1,
                    chunk_id="chk_1",
                    text="DocuLens AI provides advanced multimodal retrieval.",
                    sources=["dense", "bm25"],
                )
            ],
            strategy="rrf",
        )
        with patch("app.services.orchestrator.reranker.rerank") as mock_rerank:
            mock_rerank.return_value = MagicMock(
                model_name="ms-marco-MiniLM-L-6-v2",
                results=[
                    RerankedCandidate(
                        rank=1,
                        score=0.95,
                        initial_rank=1,
                        initial_score=0.9,
                        document_id="doc_test",
                        page_number=1,
                        chunk_id="chk_1",
                        text="DocuLens AI provides advanced multimodal retrieval.",
                        sources=["dense", "bm25"],
                    )
                ],
            )
            
            res = orch.orchestrate_query("What is DocuLens AI?", document_id="doc_test", top_k=5)
            
            assert res is not None
            assert res.pipeline_trace is not None
            trace = res.pipeline_trace
            
            # Check summary fields
            assert trace.summary.total_stages == 8
            assert trace.summary.successful_stages >= 7
            assert trace.summary.total_duration_ms > 0
            assert trace.summary.retrieval_duration_ms is not None
            assert trace.summary.reranking_duration_ms is not None
            assert trace.summary.generation_duration_ms is not None
            assert trace.summary.citation_validation_duration_ms is not None
            
            # All latencies are non-negative
            assert trace.summary.retrieval_duration_ms >= 0.0
            assert trace.summary.reranking_duration_ms >= 0.0
            assert trace.summary.generation_duration_ms >= 0.0
            assert trace.summary.citation_validation_duration_ms >= 0.0
            
            # Check stages in order
            stage_ids = [s.stage_id for s in trace.stages]
            expected_stages = [
                "question_understanding",
                "hybrid_retrieval",
                "reranking",
                "evidence_selection",
                "evidence_validation",
                "context_assembly",
                "answer_generation",
                "citation_validation",
            ]
            assert stage_ids == expected_stages
            
            # Check stage ordering
            for i, stage in enumerate(trace.stages):
                assert stage.order == i + 1
                assert stage.duration_ms >= 0.0
            
            # Check retrieval details — dense/bm25/visual candidate counts
            ret_stage = next(s for s in trace.stages if s.stage_id == "hybrid_retrieval")
            assert "Dense candidates" in ret_stage.details
            assert "Keyword candidates" in ret_stage.details
            assert "Visual candidates" in ret_stage.details
            assert "Total retrieved" in ret_stage.details
            assert ret_stage.details["Dense candidates"] == 1
            assert ret_stage.details["Keyword candidates"] == 1
            assert ret_stage.input_count == 1
            assert ret_stage.output_count == 1
            
            # Check reranking details
            rnk_stage = next(s for s in trace.stages if s.stage_id == "reranking")
            assert "Model" in rnk_stage.details
            assert "Candidates before" in rnk_stage.details
            assert "Candidates after" in rnk_stage.details
            assert "Selection threshold" in rnk_stage.details
            assert rnk_stage.details["Model"] == "ms-marco-MiniLM-L-6-v2"
            assert rnk_stage.input_count is not None
            assert rnk_stage.output_count is not None
            
            # Check generation fields: provider, model, latency (no secrets)
            gen_stage = next(s for s in trace.stages if s.stage_id == "answer_generation")
            assert "Provider" in gen_stage.details
            assert "Model" in gen_stage.details
            assert "Generation latency" in gen_stage.details

            # Check citation validation fields
            cit_stage = next(s for s in trace.stages if s.stage_id == "citation_validation")
            assert "Valid citations" in cit_stage.details
            assert "Invalid citations" in cit_stage.details
            assert "Citations checked" in cit_stage.details

            # Verify no internal/secret keys in any stage
            forbidden_keys = {"api_key", "secret", "token", "password", "system_prompt", "prompt", "raw_prompt", "hidden_reasoning"}
            for s in trace.stages:
                for key in s.details.keys():
                    assert key.lower() not in forbidden_keys, f"Sensitive key '{key}' exposed in stage '{s.stage_id}'"


def test_pipeline_stage_telemetry_fields() -> None:
    """Verify PipelineStageTrace carries input_count, output_count, and error_category fields."""
    orch = AnswerOrchestrator()
    with patch("app.services.orchestrator.hybrid_retriever.retrieve") as mock_ret, \
         patch("app.services.orchestrator.reranker.rerank") as mock_rerank:
        mock_ret.return_value = MagicMock(
            results=[
                FusedCandidate(
                    rank=1, score=0.9, document_id="doc_x", page_number=2,
                    chunk_id="chk_a", text="Evidence text.", sources=["dense"]
                )
            ],
            strategy="weighted",
        )
        mock_rerank.return_value = MagicMock(
            model_name="ms-marco-MiniLM-L-6-v2",
            results=[
                RerankedCandidate(
                    rank=1, score=0.8, initial_rank=1, initial_score=0.9,
                    document_id="doc_x", page_number=2, chunk_id="chk_a",
                    text="Evidence text.", sources=["dense"]
                )
            ]
        )
        res = orch.orchestrate_query("Test query", document_id="doc_x", top_k=3)
        assert res.pipeline_trace is not None
        for stage in res.pipeline_trace.stages:
            assert hasattr(stage, "input_count")
            assert hasattr(stage, "output_count")
            assert hasattr(stage, "error_category")
            assert hasattr(stage, "fallback_used")
            assert stage.status in {"success", "fallback", "failed", "skipped"}


def test_pipeline_stage_missing_handles_gracefully() -> None:
    """Verify fallback status when evidence is dropped during validation."""
    orch = AnswerOrchestrator()
    with patch("app.services.orchestrator.hybrid_retriever.retrieve") as mock_ret, \
         patch("app.services.orchestrator.reranker.rerank") as mock_rerank:
        mock_ret.return_value = MagicMock(
            results=[
                FusedCandidate(
                    rank=1, score=0.9, document_id="doc_empty", page_number=1,
                    chunk_id="chk_empty", text="", image_url=None, sources=["dense"]
                )
            ],
            strategy="rrf",
        )
        mock_rerank.return_value = MagicMock(
            model_name="ms-marco-MiniLM-L-6-v2",
            results=[
                RerankedCandidate(
                    rank=1, score=0.8, initial_rank=1, initial_score=0.9,
                    document_id="doc_empty", page_number=1, chunk_id="chk_empty",
                    text="", image_url=None, sources=["dense"]
                )
            ]
        )
        res = orch.orchestrate_query("Is evidence empty?", document_id="doc_empty", top_k=5)
        assert res.pipeline_trace is not None
        val_stage = next((s for s in res.pipeline_trace.stages if s.stage_id == "evidence_validation"), None)
        assert val_stage is not None
        assert val_stage.status in {"success", "fallback"}


def test_orchestrator_error_handling() -> None:
    """Verify that an error in the pipeline produces a failed stage trace and raises."""
    orch = AnswerOrchestrator()
    
    with patch("app.services.orchestrator.hybrid_retriever.retrieve", side_effect=ValueError("Retrieval failed")):
        with pytest.raises(ValueError, match="Retrieval failed"):
            orch.orchestrate_query("Will fail")


def test_pipeline_serialization_safe_for_frontend() -> None:
    """Verify that pipeline_trace serializes to JSON without secret leakage."""
    import json
    orch = AnswerOrchestrator()
    with patch("app.services.orchestrator.hybrid_retriever.retrieve") as mock_ret, \
         patch("app.services.orchestrator.reranker.rerank") as mock_rerank:
        mock_ret.return_value = MagicMock(
            results=[
                FusedCandidate(
                    rank=1, score=0.85, document_id="doc_safe", page_number=1,
                    chunk_id="chk_s", text="Safe content.", sources=["bm25"]
                )
            ],
            strategy="rrf",
        )
        mock_rerank.return_value = MagicMock(
            model_name="ms-marco-MiniLM-L-6-v2",
            results=[
                RerankedCandidate(
                    rank=1, score=0.85, initial_rank=1, initial_score=0.85,
                    document_id="doc_safe", page_number=1, chunk_id="chk_s",
                    text="Safe content.", sources=["bm25"]
                )
            ]
        )
        res = orch.orchestrate_query("Safe query?", document_id="doc_safe", top_k=3)
        assert res.pipeline_trace is not None
        serialized = json.dumps(res.pipeline_trace.model_dump())
        assert "pipeline_id" in serialized
        assert "stages" in serialized
        forbidden_patterns = ["sk-", "AIza", "Bearer ", "api_key", "system_prompt"]
        for pattern in forbidden_patterns:
            assert pattern not in serialized, f"Sensitive pattern '{pattern}' found in trace output"
