import logging
import time
import uuid
from typing import Any, Literal
from app.core.logger import sanitize_structured_data
from app.schemas.generation import GenerationResult, PipelineStageTrace, PipelineSummary, PipelineTrace
from app.services.hybrid import hybrid_retriever
from app.services.reranker import reranker
from app.services.evidence import evidence_selector
from app.services.context_assembler import context_assembler
from app.services.generator import generator
from app.services.citation_validator import citation_validator

logger = logging.getLogger(__name__)


class AnswerOrchestrator:
    def orchestrate_query(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> GenerationResult:
        trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        query_id = f"q_{uuid.uuid4().hex[:12]}"
        stages: list[PipelineStageTrace] = []
        t_pipeline_start = time.perf_counter()
        
        def add_stage(
            sid: str,
            name: str,
            status: Literal["success", "fallback", "failed", "skipped"],
            t0: float,
            t1: float,
            desc: str,
            details: dict[str, Any],
            input_count: int | None = None,
            output_count: int | None = None,
            fallback: bool = False,
            f_reason: str | None = None,
            err_category: str | None = None,
            err_msg: str | None = None,
        ) -> None:
            duration_ms = max(0.0, (t1 - t0) * 1000)
            safe_details = sanitize_structured_data(details)
            stage_trace = PipelineStageTrace(
                stage_id=sid,
                stage_name=name,
                order=len(stages) + 1,
                status=status,
                duration_ms=duration_ms,
                input_count=input_count,
                output_count=output_count,
                error_category=err_category,
                description=desc,
                fallback_used=fallback,
                fallback_reason=f_reason,
                error_message=err_msg,
                details=safe_details,
            )
            stages.append(stage_trace)
            
            logger_level = logging.ERROR if status == "failed" else (logging.WARNING if status == "fallback" else logging.INFO)
            logger.log(
                logger_level,
                f"Pipeline stage '{name}' completed with status '{status}'",
                extra={
                    "operation": "pipeline",
                    "query_id": query_id,
                    "stage": sid,
                    "status": status,
                    "duration_ms": round(duration_ms, 2),
                    "document_id": document_id,
                    "error_type": err_msg,
                },
            )

        # Stage 1: Question Understanding
        t0 = time.perf_counter()
        clean_question = question.strip() if question else ""
        t1 = time.perf_counter()
        add_stage(
            sid="question_understanding",
            name="Question understanding",
            status="success",
            t0=t0,
            t1=t1,
            desc=f"Query parsed ({len(clean_question)} chars)",
            details={
                "Query length": f"{len(clean_question)} chars",
                "Target document": document_id or "All documents",
            },
            input_count=len(question) if question else 0,
            output_count=len(clean_question),
        )

        retrieval_dur: float | None = None
        rerank_dur: float | None = None
        gen_dur: float | None = None
        citation_val_dur: float | None = None

        try:
            # Stage 2: Hybrid Retrieval
            t0 = time.perf_counter()
            hybrid_res = hybrid_retriever.retrieve(
                query=clean_question,
                top_k=20,
                document_id=document_id,
                score_threshold=score_threshold,
            )
            t1 = time.perf_counter()
            retrieval_dur = max(0.0, (t1 - t0) * 1000)
            dense_count = sum(1 for c in hybrid_res.results if "dense" in c.sources)
            bm25_count = sum(1 for c in hybrid_res.results if "bm25" in c.sources)
            visual_count = sum(1 for c in hybrid_res.results if "visual" in c.sources)
            add_stage(
                sid="hybrid_retrieval",
                name="Hybrid retrieval",
                status="success",
                t0=t0,
                t1=t1,
                desc=f"Retrieved {len(hybrid_res.results)} candidates across channels",
                details={
                    "Dense candidates": dense_count,
                    "Keyword candidates": bm25_count,
                    "Visual candidates": visual_count,
                    "Total retrieved": len(hybrid_res.results),
                    "Fusion strategy": hybrid_res.strategy,
                },
                input_count=1,
                output_count=len(hybrid_res.results),
            )

            # Stage 3: Reranking
            t0 = time.perf_counter()
            rerank_fallback = False
            rerank_reason: str | None = None
            rerank_err_cat: str | None = None
            rerank_candidates: list[Any] = []
            model_name = getattr(reranker, "model_name", "cross-encoder")

            try:
                rerank_res = reranker.rerank(
                    query=clean_question,
                    candidates=hybrid_res.results,
                    top_k=10,
                    target_document_id=document_id,
                    score_threshold=score_threshold,
                )
                rerank_candidates = rerank_res.results
                model_name = rerank_res.model_name
            except Exception as exc:
                logger.warning(
                    f"Reranking failed: {exc}. Falling back to unreranked hybrid retrieval results.",
                    extra={"query": clean_question, "document_id": document_id, "error": str(exc)},
                )
                rerank_fallback = True
                rerank_reason = f"Reranker failed ({type(exc).__name__}), fell back to hybrid retrieval results"
                rerank_err_cat = "reranker_failure"
                rerank_candidates = list(hybrid_res.results)

            t1 = time.perf_counter()
            rerank_dur = max(0.0, (t1 - t0) * 1000)
            add_stage(
                sid="reranking",
                name="Reranking",
                status="success" if not rerank_fallback else "fallback",
                t0=t0,
                t1=t1,
                desc=f"Cross-encoder scored {len(rerank_candidates)} candidates" if not rerank_fallback else f"Reranker unavailable, fell back to {len(rerank_candidates)} hybrid candidates",
                details={
                    "Model": model_name,
                    "Candidates before": len(hybrid_res.results),
                    "Candidates after": len(rerank_candidates),
                    "Selection threshold": score_threshold if score_threshold is not None else "None",
                },
                input_count=len(hybrid_res.results),
                output_count=len(rerank_candidates),
                fallback=rerank_fallback,
                f_reason=rerank_reason,
                err_category=rerank_err_cat,
            )

            # Stage 4: Evidence Selection
            t0 = time.perf_counter()
            selection_res = evidence_selector.select_evidence(
                query=clean_question,
                candidates=rerank_candidates,
                top_k=top_k,
                target_document_id=document_id,
                score_threshold=score_threshold,
            )
            t1 = time.perf_counter()
            add_stage(
                sid="evidence_selection",
                name="Evidence selection",
                status="success",
                t0=t0,
                t1=t1,
                desc=f"Selected {len(selection_res.evidence)} evidence items (top-k={top_k})",
                details={
                    "Candidates input": len(rerank_candidates),
                    "Selected count": len(selection_res.evidence),
                    "Top-k target": top_k,
                    "Deduplicated": selection_res.total_candidates - len(selection_res.evidence),
                    "Threshold applied": score_threshold if score_threshold is not None else "None",
                },
                input_count=len(rerank_candidates),
                output_count=len(selection_res.evidence),
            )

            # Stage 5: Evidence Validation
            t0 = time.perf_counter()
            valid_items = []
            for ev in selection_res.evidence:
                has_content = bool(ev.text and ev.text.strip()) or bool(ev.image_url and ev.image_url.strip())
                doc_matches = (document_id is None) or (ev.document_id == document_id)
                if has_content and doc_matches:
                    valid_items.append(ev)

            dropped_count = len(selection_res.evidence) - len(valid_items)
            validation_fallback = dropped_count > 0
            t1 = time.perf_counter()
            add_stage(
                sid="evidence_validation",
                name="Evidence validation",
                status="success" if not validation_fallback else "fallback",
                t0=t0,
                t1=t1,
                desc=f"Validated {len(valid_items)} of {len(selection_res.evidence)} evidence items",
                details={
                    "Input evidence": len(selection_res.evidence),
                    "Valid items": len(valid_items),
                    "Rejected items": dropped_count,
                    "Validation status": "Passed" if dropped_count == 0 else "Degraded",
                },
                input_count=len(selection_res.evidence),
                output_count=len(valid_items),
                fallback=validation_fallback,
                f_reason="Corrupt/isolated evidence dropped" if validation_fallback else None,
                err_category="invalid_evidence" if validation_fallback else None,
            )

            # Stage 6: Context Assembly
            t0 = time.perf_counter()
            context_res = context_assembler.assemble_context(
                evidence=valid_items,
                max_chars=4000,
            )
            t1 = time.perf_counter()
            distinct_pages = len(set(e.page_number for e in valid_items))
            add_stage(
                sid="context_assembly",
                name="Context assembly",
                status="success",
                t0=t0,
                t1=t1,
                desc=f"Assembled context ({len(context_res.context_text)} chars, {distinct_pages} pages)",
                details={
                    "Evidence count": len(valid_items),
                    "Context size": f"{len(context_res.context_text)} chars",
                    "Source pages": f"{distinct_pages} page(s)",
                    "Truncated": "Yes" if context_res.truncated else "No",
                },
                input_count=len(valid_items),
                output_count=len(context_res.context_text),
            )

            # Stage 7: Answer Generation
            t0 = time.perf_counter()
            gen_res = generator.generate_answer(
                question=clean_question,
                evidence=valid_items,
                document_id=document_id,
            )
            t1 = time.perf_counter()
            gen_dur = max(0.0, (t1 - t0) * 1000)
            gen_details: dict[str, Any] = {
                "Provider": gen_res.provider,
                "Model": gen_res.model,
                "Generation latency": f"{gen_dur:.0f} ms",
            }
            if gen_res.usage:
                if "prompt_tokens" in gen_res.usage:
                    gen_details["Prompt tokens"] = gen_res.usage["prompt_tokens"]
                if "completion_tokens" in gen_res.usage:
                    gen_details["Completion tokens"] = gen_res.usage["completion_tokens"]
                if "total_tokens" in gen_res.usage:
                    gen_details["Total tokens"] = gen_res.usage["total_tokens"]

            add_stage(
                sid="answer_generation",
                name="Answer generation",
                status="success",
                t0=t0,
                t1=t1,
                desc=f"Answer generated using {gen_res.model}",
                details=gen_details,
                input_count=len(valid_items),
                output_count=len(gen_res.answer),
            )

            # Stage 8: Citation Validation
            t0 = time.perf_counter()
            val_res = citation_validator.validate_citations(
                answer=gen_res.answer,
                evidence=valid_items,
                target_document_id=document_id,
            )
            t1 = time.perf_counter()
            citation_val_dur = max(0.0, (t1 - t0) * 1000)
            total_citations_checked = val_res.valid_count + val_res.invalid_count
            citation_fallback = val_res.invalid_count > 0
            add_stage(
                sid="citation_validation",
                name="Citation validation",
                status="success" if not citation_fallback else "fallback",
                t0=t0,
                t1=t1,
                desc=f"{val_res.valid_count} valid, {val_res.invalid_count} rejected",
                details={
                    "Citations checked": total_citations_checked,
                    "Valid citations": val_res.valid_count,
                    "Invalid citations": val_res.invalid_count,
                    "Fallback status": "Fallback applied" if citation_fallback else "None (Strict)",
                },
                input_count=total_citations_checked,
                output_count=val_res.valid_count,
                fallback=citation_fallback,
                f_reason="Invalid citations dropped" if citation_fallback else None,
                err_category="invalid_citations" if citation_fallback else None,
            )

            total_dur = max(0.0, (time.perf_counter() - t_pipeline_start) * 1000)
            gen_res.pipeline_trace = PipelineTrace(
                pipeline_id=trace_id,
                stages=stages,
                summary=PipelineSummary(
                    total_duration_ms=total_dur,
                    retrieval_duration_ms=retrieval_dur,
                    reranking_duration_ms=rerank_dur,
                    generation_duration_ms=gen_dur,
                    citation_validation_duration_ms=citation_val_dur,
                    total_stages=len(stages),
                    successful_stages=sum(1 for s in stages if s.status == "success"),
                    failed_stages=sum(1 for s in stages if s.status == "failed"),
                    fallback_stages=sum(1 for s in stages if s.status == "fallback"),
                ),
            )

            from app.core.metrics import metrics_collector
            metrics_collector.record_retrieval(
                duration_ms=retrieval_dur or 0.0,
                result_count=len(hybrid_res.results) if hybrid_res else 0,
            )
            metrics_collector.record_reranking(
                duration_ms=rerank_dur or 0.0,
            )
            metrics_collector.record_generation(
                duration_ms=gen_dur or 0.0,
                is_grounded=gen_res.is_grounded,
                valid_citations=val_res.valid_count if val_res else 0,
                rejected_citations=val_res.invalid_count if val_res else 0,
                is_refusal=not gen_res.is_grounded,
            )

            return gen_res
        except Exception as e:
            t1 = time.perf_counter()
            from app.core.metrics import metrics_collector
            from app.services.llm_provider import ProviderRateLimitError, ProviderTimeoutError
            if isinstance(e, ProviderTimeoutError):
                metrics_collector.record_provider_timeout()
            elif isinstance(e, ProviderRateLimitError):
                metrics_collector.record_provider_rate_limit()

            add_stage(
                sid="pipeline_error",
                name="Pipeline execution",
                status="failed",
                t0=t0,
                t1=t1,
                desc="The pipeline encountered a critical error",
                details={"Error": type(e).__name__, "Message": str(e)},
                input_count=None,
                output_count=0,
                err_category=type(e).__name__,
                err_msg="Stage failed",
            )
            raise e


orchestrator = AnswerOrchestrator()
