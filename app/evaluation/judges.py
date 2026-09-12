"""Generation evaluation judges for DocuLens AI — Phase 6.4.

Provides deterministic and LLM-as-judge evaluation interfaces for assessing
faithfulness, answer relevancy, citation validity, and abstention correctness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import json
import re
from typing import Any

from app.evaluation.metrics import (
    compute_abstention_correctness,
    compute_answer_relevancy,
    compute_citation_precision,
    compute_citation_recall,
    compute_faithfulness,
)
from app.schemas.citation import Citation
from app.schemas.evaluation import GenerationEvalMetrics, GoldQuery
from app.services.citation_validator import citation_validator, is_refusal_response
from app.services.llm_provider import BaseLLMProvider


class BaseGenerationJudge(ABC):
    """Abstract interface for generation evaluation judges."""

    @abstractmethod
    def evaluate(
        self,
        query: GoldQuery,
        answer: str,
        evidence: Sequence[Any],
        citations: Sequence[Citation] | None = None,
    ) -> GenerationEvalMetrics:
        """Evaluate generated answer against query, evidence, and ground truth."""
        raise NotImplementedError


class DeterministicGenerationJudge(BaseGenerationJudge):
    """Deterministic, standard-library generation evaluation judge."""

    def evaluate(
        self,
        query: GoldQuery,
        answer: str,
        evidence: Sequence[Any],
        citations: Sequence[Citation] | None = None,
    ) -> GenerationEvalMetrics:
        is_refusal = is_refusal_response(answer)

        if citations is None:
            val_res = citation_validator.validate_citations(
                answer=answer,
                evidence=evidence,
                target_document_id=query.document_id,
            )
            eval_citations: Sequence[Any] = val_res.citations
        else:
            eval_citations = citations

        evidence_pages: list[int] = []
        for item in evidence:
            page = getattr(item, "page_number", None)
            if page is None and isinstance(item, dict):
                page = item.get("page_number")
            if page is not None and isinstance(page, int):
                evidence_pages.append(page)

        faithfulness = compute_faithfulness(
            answer=answer,
            evidence=evidence,
            question=query.question,
            is_refusal=is_refusal,
        )

        relevancy = compute_answer_relevancy(
            question=query.question,
            answer=answer,
            ground_truth_answer=query.ground_truth_answer,
            is_answerable=query.is_answerable,
            is_refusal=is_refusal,
        )

        cit_precision = compute_citation_precision(
            citations=eval_citations,
            valid_evidence_pages=evidence_pages if evidence_pages else None,
            target_document_id=query.document_id,
            is_refusal=is_refusal,
        )

        cit_recall = compute_citation_recall(
            citations=eval_citations,
            ground_truth_pages=query.ground_truth_pages,
            target_document_id=query.document_id,
        )

        abstention_correct = compute_abstention_correctness(
            answer=answer,
            is_answerable=query.is_answerable,
        )

        return GenerationEvalMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            citation_precision=cit_precision,
            citation_recall=cit_recall,
            abstention_correct=abstention_correct,
        )


class MockLLMGenerationJudge(BaseGenerationJudge):
    """Mock judge for deterministic testing with injected metric values."""

    def __init__(
        self,
        faithfulness: float = 1.0,
        answer_relevancy: float = 1.0,
        citation_precision: float = 1.0,
        citation_recall: float = 1.0,
        abstention_correct: bool = True,
        query_overrides: dict[str, GenerationEvalMetrics] | None = None,
    ) -> None:
        self.faithfulness = faithfulness
        self.answer_relevancy = answer_relevancy
        self.citation_precision = citation_precision
        self.citation_recall = citation_recall
        self.abstention_correct = abstention_correct
        self.query_overrides = query_overrides or {}

    def evaluate(
        self,
        query: GoldQuery,
        answer: str,
        evidence: Sequence[Any],
        citations: Sequence[Citation] | None = None,
    ) -> GenerationEvalMetrics:
        if query.query_id in self.query_overrides:
            return self.query_overrides[query.query_id]

        return GenerationEvalMetrics(
            faithfulness=self.faithfulness,
            answer_relevancy=self.answer_relevancy,
            citation_precision=self.citation_precision,
            citation_recall=self.citation_recall,
            abstention_correct=self.abstention_correct,
        )


class LLMJudgeEvaluator(BaseGenerationJudge):
    """LLM-as-judge evaluator adapter wrapping BaseLLMProvider."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback_judge: BaseGenerationJudge | None = None,
    ) -> None:
        self.provider = provider
        self.fallback_judge = fallback_judge or DeterministicGenerationJudge()

    def evaluate(
        self,
        query: GoldQuery,
        answer: str,
        evidence: Sequence[Any],
        citations: Sequence[Citation] | None = None,
    ) -> GenerationEvalMetrics:
        if is_refusal_response(answer) or not query.is_answerable:
            return self.fallback_judge.evaluate(
                query=query, answer=answer, evidence=evidence, citations=citations
            )

        context_snippets = []
        for idx, item in enumerate(evidence[:5], start=1):
            text = getattr(item, "text", None) or (
                item.get("text") if isinstance(item, dict) else ""
            )
            page = getattr(item, "page_number", 1) or (
                item.get("page_number", 1) if isinstance(item, dict) else 1
            )
            context_snippets.append(f"[Evidence {idx} | Page {page}]: {text}")
        context_str = "\n".join(context_snippets)

        prompt = (
            "Evaluate the generated answer on two dimensions:\n"
            "1. Faithfulness (0.0 to 1.0): Are all factual claims supported by context?\n"
            "2. Answer Relevancy (0.0 to 1.0): Does answer address the question?\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{query.question}\n\n"
            f"Reference Answer:\n{query.ground_truth_answer}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            'Respond ONLY with JSON: {"faithfulness": 1.0, "answer_relevancy": 1.0}'
        )

        try:
            response = self.provider.generate(
                prompt=prompt,
                system_prompt="You are a strict evaluation judge. Output only JSON.",
                temperature=0.0,
            )
            raw_text = (
                response.content.strip()
                if hasattr(response, "content")
                else getattr(response, "text", str(response)).strip()
            )
            json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                faithfulness = max(0.0, min(1.0, float(data.get("faithfulness", 1.0))))
                relevancy = max(0.0, min(1.0, float(data.get("answer_relevancy", 1.0))))
            else:
                return self.fallback_judge.evaluate(
                    query=query, answer=answer, evidence=evidence, citations=citations
                )

            det_metrics = self.fallback_judge.evaluate(
                query=query, answer=answer, evidence=evidence, citations=citations
            )

            return GenerationEvalMetrics(
                faithfulness=round(faithfulness, 6),
                answer_relevancy=round(relevancy, 6),
                citation_precision=det_metrics.citation_precision,
                citation_recall=det_metrics.citation_recall,
                abstention_correct=det_metrics.abstention_correct,
            )
        except Exception:
            return self.fallback_judge.evaluate(
                query=query, answer=answer, evidence=evidence, citations=citations
            )
