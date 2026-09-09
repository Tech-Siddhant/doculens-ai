import re
from typing import Any

from app.schemas.generation import Citation, GenerationResult
from app.schemas.retrieval import RetrievedChunk
from app.services.llm_provider import BaseLLMProvider, llm_provider

DEFAULT_SYSTEM_PROMPT = (
    "You are DocuLens AI, a document question-answering assistant.\n"
    "Your task is to answer user questions using ONLY the provided evidence excerpts."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I do not have sufficient information in the provided document to answer this question."
)

REFUSAL_PHRASES = (
    INSUFFICIENT_EVIDENCE_ANSWER.lower(),
    "insufficient information",
    "insufficient evidence",
    "i do not have sufficient information",
    "does not contain sufficient information",
    "no information provided",
    "not mentioned in the provided",
)


def build_grounding_prompt(question: str, evidence: list[RetrievedChunk]) -> str:
    """Constructs a structured grounding prompt strictly separating evidence from question."""
    evidence_lines: list[str] = []
    for idx, chunk in enumerate(evidence, start=1):
        header = (
            f"[Evidence {idx}] Document: {chunk.document_id} | "
            f"Page: {chunk.page_number} | "
            f"Chunk ID: {chunk.chunk_id} | "
            f"Score: {chunk.score:.4f}"
        )
        evidence_lines.append(f"{header}\n{chunk.text.strip()}")

    evidence_block = "\n\n".join(evidence_lines)

    return (
        "STRICT GROUNDING RULES:\n"
        "1. Base your answer ONLY on the provided evidence excerpts below. "
        "Do NOT assume, speculate, or bring in outside knowledge.\n"
        "2. Cite evidence using evidence reference tags (e.g. [Evidence 1] or [Evidence 2]) directly after claims derived from that evidence.\n"
        "3. If the provided evidence does not contain sufficient information to answer the question, "
        f'reply: "{INSUFFICIENT_EVIDENCE_ANSWER}"\n'
        "4. Keep the answer direct, factual, and concise.\n\n"
        "--- EVIDENCE ---\n"
        f"{evidence_block}\n"
        "--- END EVIDENCE ---\n\n"
        "--- USER QUESTION ---\n"
        f"{question.strip()}\n"
        "--- END USER QUESTION ---\n\n"
        "Answer:"
    )


def is_refusal_response(answer: str) -> bool:
    """Detect whether generated text is an explicit refusal or indicates insufficient evidence."""
    clean = answer.strip().lower()
    return any(phrase in clean for phrase in REFUSAL_PHRASES)


def extract_citation_indices(text: str) -> list[int]:
    """Extract referenced 1-based evidence indices from text in order of appearance.

    Supports formats:
    - [Evidence 1], [evidence 2]
    - Numeric bracket references: [1], [2]
    """
    pattern = re.compile(r"\[(?:Evidence\s+)?(\d+)\]", re.IGNORECASE)
    indices: list[int] = []
    seen: set[int] = set()
    for match in pattern.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def make_citation(index: int, chunk: RetrievedChunk) -> Citation:
    """Build a validated Citation object from a RetrievedChunk."""
    return Citation(
        reference=f"[Evidence {index}]",
        rank=chunk.rank,
        score=chunk.score,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        evidence_text=chunk.text,
        text=chunk.text,
        metadata=chunk.metadata,
    )


def validate_and_build_citations(
    answer: str, evidence: list[RetrievedChunk]
) -> tuple[list[Citation], bool]:
    """Validate citations against retrieved evidence deterministically.

    Returns:
        tuple[list[Citation], bool]: (citations, is_grounded)
    """
    if not evidence or is_refusal_response(answer):
        return [], False

    evidence_map = {idx: chunk for idx, chunk in enumerate(evidence, start=1)}
    cited_indices = extract_citation_indices(answer)

    if cited_indices:
        # Filter valid references and reject out-of-range/fabricated references
        valid_citations = [
            make_citation(idx, evidence_map[idx])
            for idx in cited_indices
            if idx in evidence_map
        ]
        # If every citation tag was fabricated/out-of-range, answer is not grounded
        if not valid_citations:
            return [], False
        return valid_citations, True

    # If no explicit citation tags appeared and answer is not a refusal, map retrieved chunks
    all_citations = [
        make_citation(idx, chunk) for idx, chunk in enumerate(evidence, start=1)
    ]
    return all_citations, True


class AnswerGenerator:
    """Service for generating grounded answers using retrieved evidence chunks."""

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or llm_provider

    def generate_answer(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        document_id: str | None = None,
    ) -> GenerationResult:
        """Generate a grounded answer and citation metadata from evidence chunks."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list of RetrievedChunk objects.")

        clean_question = question.strip()

        # Handle empty evidence immediately without calling LLM provider
        if len(evidence) == 0:
            return GenerationResult(
                question=clean_question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                document_id=document_id,
                model="none",
                provider="rule_based",
                citations=[],
                is_grounded=False,
                usage={},
            )

        prompt = build_grounding_prompt(clean_question, evidence)
        llm_response = self.provider.generate(
            prompt=prompt, system_prompt=DEFAULT_SYSTEM_PROMPT
        )

        citations, is_grounded = validate_and_build_citations(
            llm_response.content, evidence
        )

        return GenerationResult(
            question=clean_question,
            answer=llm_response.content,
            document_id=document_id,
            model=llm_response.model,
            provider=llm_response.provider,
            citations=citations,
            is_grounded=is_grounded,
            usage=llm_response.usage,
        )


generator = AnswerGenerator()
