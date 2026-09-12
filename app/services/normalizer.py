"""Score normalization utilities for multi-channel retrieval fusion."""

import copy
import math
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def min_max_scale_scores(scores: Sequence[float]) -> list[float]:
    """Normalize a sequence of raw retrieval scores into [0.0, 1.0] using Min-Max scaling.

    Semantics:
    - Higher raw score indicates higher relevance across all retrievers (dense, BM25, visual).
    - If all scores are identical:
        - If score > 0.0, normalized score is 1.0 (equally top-relevant).
        - If score <= 0.0, normalized score is 0.0 (zero/negative relevance).
    - If max_score > min_score:
        - score_norm = (score - min_score) / (max_score - min_score)
    - Numerically stable for extreme ranges and safe against division by zero.
    """
    if not scores:
        return []

    float_scores = [float(s) for s in scores]
    s_min = min(float_scores)
    s_max = max(float_scores)

    # Safe equality check to prevent floating point division by zero
    if math.isclose(s_max, s_min, rel_tol=1e-9, abs_tol=1e-12):
        constant_val = 1.0 if s_max > 0.0 else 0.0
        return [constant_val for _ in float_scores]

    denom = s_max - s_min
    return [round((s - s_min) / denom, 6) for s in float_scores]


def normalize_retrieval_results(
    results: Sequence[T],
    score_attr: str = "score",
) -> list[T]:
    """Normalize retrieval candidate scores without mutating the original input objects.

    Sets:
    - raw_score: original unnormalized retrieval score
    - normalized_score: min-max normalized score in [0.0, 1.0]

    Preserves candidate ranking, document_id, page_number, chunk_id, and metadata.
    Supports Pydantic models (e.g. RetrievedChunk, RetrievedVisualPage) and dictionaries.
    """
    if not results:
        return []

    # Extract raw scores
    raw_scores: list[float] = []
    for item in results:
        if isinstance(item, dict):
            raw_s = item.get("raw_score", item.get(score_attr, 0.0))
        elif isinstance(item, BaseModel):
            raw_s = getattr(item, "raw_score", None)
            if raw_s is None:
                raw_s = getattr(item, score_attr, 0.0)
        else:
            raw_s = getattr(item, "raw_score", getattr(item, score_attr, 0.0))
        raw_scores.append(float(raw_s))

    normalized_scores = min_max_scale_scores(raw_scores)

    # Construct new immutable objects
    normalized_results: list[T] = []
    for item, raw_s, norm_s in zip(results, raw_scores, normalized_scores, strict=True):
        if isinstance(item, BaseModel):
            updated_item = item.model_copy(
                update={"raw_score": raw_s, "normalized_score": norm_s}
            )
            normalized_results.append(updated_item)  # type: ignore[arg-type]
        elif isinstance(item, dict):
            dict_copy = copy.deepcopy(item)
            dict_copy["raw_score"] = raw_s
            dict_copy["normalized_score"] = norm_s
            normalized_results.append(dict_copy)  # type: ignore[arg-type]
        else:
            obj_copy = copy.copy(item)
            setattr(obj_copy, "raw_score", raw_s)
            setattr(obj_copy, "normalized_score", norm_s)
            normalized_results.append(obj_copy)

    return normalized_results
