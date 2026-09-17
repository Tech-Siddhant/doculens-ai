"""Evaluation harness and store lifecycle management for DocuLens AI — Phase 8.7.

Ensures strict object identity and lifecycle consistency between indexing and retrieval,
and defines standard, reproducible evaluation baseline configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from app.evaluation.runner import RetrievalEvaluator
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExperimentResult,
    GoldQuery,
    PipelineConfig,
)
from app.services.bm25 import BM25Retriever, bm25_retriever as singleton_bm25
from app.services.chunker import chunk_extraction_result
from app.services.dataset import GoldBenchmark
from app.services.embedder import embedding_service
from app.services.extractor import extract_text_and_metadata
from app.services.hybrid import HybridRetriever, hybrid_retriever as singleton_hybrid
from app.services.reranker import CrossEncoderReranker, reranker as singleton_reranker
from app.services.retriever import DenseRetriever, retriever as singleton_dense
from app.services.vector_store import QdrantVectorStore, vector_store as singleton_vstore
from app.services.visual_vector_store import (
    VisualQdrantVectorStore,
    visual_vector_store as singleton_vis_vstore,
)


@dataclass
class EvaluationStores:
    """Explicit container guaranteeing that indexing and retrieval share the exact same instances."""

    vector_store: QdrantVectorStore
    bm25_retriever: BM25Retriever
    visual_vector_store: VisualQdrantVectorStore
    dense_retriever: DenseRetriever
    hybrid_retriever: HybridRetriever
    reranker: CrossEncoderReranker


def get_shared_stores() -> EvaluationStores:
    """Retrieve the application's shared singleton store instances."""
    return EvaluationStores(
        vector_store=singleton_vstore,
        bm25_retriever=singleton_bm25,
        visual_vector_store=singleton_vis_vstore,
        dense_retriever=singleton_dense,
        hybrid_retriever=singleton_hybrid,
        reranker=singleton_reranker,
    )


def create_isolated_stores() -> EvaluationStores:
    """Instantiate a completely isolated, fresh in-memory store suite for testing."""
    vstore = QdrantVectorStore(location=":memory:", collection_name="eval_text_chunks")
    bm25 = BM25Retriever()
    vis_store = VisualQdrantVectorStore(
        location=":memory:", collection_name="eval_visual_pages"
    )
    dense_ret = DenseRetriever(embedder=embedding_service, store=vstore)
    hybrid_ret = HybridRetriever(
        dense_retriever_inst=dense_ret,
        bm25_retriever_inst=bm25,
        visual_store_inst=vis_store,
    )
    return EvaluationStores(
        vector_store=vstore,
        bm25_retriever=bm25,
        visual_vector_store=vis_store,
        dense_retriever=dense_ret,
        hybrid_retriever=hybrid_ret,
        reranker=singleton_reranker,
    )


def index_evaluation_document(
    stores: EvaluationStores,
    document_id: str,
    pdf_path: Path | str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict[str, int]:
    """Index an evaluation document into the stores, ensuring identical object identity."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation PDF not found: {path}")

    extraction = extract_text_and_metadata(document_id, path)
    chunking = chunk_extraction_result(
        extraction, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    embedded = embedding_service.embed_chunks(chunking.chunks)
    stores.vector_store.delete_by_document(document_id)
    dense_count = stores.vector_store.upsert_chunks(embedded)

    stores.bm25_retriever.delete_by_document(document_id)
    bm25_count = stores.bm25_retriever.index_chunks(chunking.chunks)

    # Multimodal Visual indexing
    try:
        import pymupdf
        from app.core.config import settings
        from app.schemas.document import RenderedPage
        from app.services.visual_embedder import visual_embedding_service

        doc = pymupdf.open(str(path))
        temp_dir = Path(settings.RENDER_OUTPUT_DIR) / "eval_renders" / document_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        rendered_pages = []
        for page_idx in range(len(doc)):
            p = doc[page_idx]
            pix = p.get_pixmap(dpi=100)
            img_file = temp_dir / f"{document_id}_p{page_idx + 1}.png"
            pix.save(str(img_file))
            rendered_pages.append(
                RenderedPage(
                    document_id=document_id,
                    page_number=page_idx + 1,
                    image_path=str(img_file),
                    width=pix.width,
                    height=pix.height,
                    size_bytes=img_file.stat().st_size,
                    format="png",
                )
            )
        doc.close()
        if rendered_pages:
            embedded_pages = visual_embedding_service.embed_rendered_pages(rendered_pages)
            stores.visual_vector_store.delete_by_document(document_id)
            vis_count = stores.visual_vector_store.upsert_pages(embedded_pages)
        else:
            vis_count = 0
    except Exception:
        vis_count = 0

    return {"dense_chunks": dense_count, "bm25_chunks": bm25_count, "visual_pages": vis_count}


def index_evaluation_corpus(
    stores: EvaluationStores,
    corpus_dir: Path | str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict[str, dict[str, int]]:
    """Index all PDF documents found in the corpus directory into the evaluation stores."""
    dir_path = Path(corpus_dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {dir_path}")

    stats: dict[str, dict[str, int]] = {}
    pdf_files = sorted(dir_path.glob("*.pdf"))
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        stats[doc_id] = index_evaluation_document(
            stores=stores,
            document_id=doc_id,
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return stats



BASELINE_NAMES = ["dense", "dense_bm25", "dense_bm25_visual", "hybrid_reranked"]


def get_retrieval_function_for_baseline(
    stores: EvaluationStores,
    baseline_name: str,
    top_k: int = 5,
) -> tuple[PipelineConfig, Callable[[GoldQuery], list[Any]]]:
    """Build retrieval function and PipelineConfig for a designated named baseline."""
    if baseline_name == "dense":
        config = PipelineConfig(
            name="dense",
            retrieval_types=["dense"],
            weights={"dense": 1.0, "bm25": 0.0, "visual": 0.0},
            reranker_enabled=False,
            top_k_retrieve=top_k,
        )

        def retrieve_fn(q: GoldQuery) -> list[Any]:
            return stores.dense_retriever.retrieve(
                query=q.question,
                top_k=top_k,
                document_id=q.document_id or None,
            ).results

        return config, retrieve_fn

    if baseline_name == "dense_bm25":
        config = PipelineConfig(
            name="dense_bm25",
            retrieval_types=["dense", "bm25"],
            weights={"dense": 0.7, "bm25": 0.3, "visual": 0.0},
            reranker_enabled=False,
            top_k_retrieve=top_k,
        )

        def retrieve_fn(q: GoldQuery) -> list[Any]:
            return stores.hybrid_retriever.retrieve(
                query=q.question,
                top_k=top_k,
                document_id=q.document_id or None,
                strategy="weighted",
                weights={"dense": 0.7, "bm25": 0.3, "visual": 0.0},
                include_dense=True,
                include_bm25=True,
                include_visual=False,
            ).results

        return config, retrieve_fn

    if baseline_name == "dense_bm25_visual":
        config = PipelineConfig(
            name="dense_bm25_visual",
            retrieval_types=["dense", "bm25", "visual"],
            weights={"dense": 0.5, "bm25": 0.3, "visual": 0.2},
            reranker_enabled=False,
            top_k_retrieve=top_k,
        )

        def retrieve_fn(q: GoldQuery) -> list[Any]:
            return stores.hybrid_retriever.retrieve(
                query=q.question,
                top_k=top_k,
                document_id=q.document_id or None,
                strategy="weighted",
                weights={"dense": 0.5, "bm25": 0.3, "visual": 0.2},
                include_dense=True,
                include_bm25=True,
                include_visual=True,
            ).results

        return config, retrieve_fn

    if baseline_name == "hybrid_reranked":
        config = PipelineConfig(
            name="hybrid_reranked",
            retrieval_types=["dense", "bm25", "visual"],
            weights={"dense": 0.5, "bm25": 0.3, "visual": 0.2},
            reranker_enabled=True,
            top_k_retrieve=max(top_k * 2, 10),
            top_k_rerank=top_k,
        )

        def retrieve_fn(q: GoldQuery) -> list[Any]:
            candidates = stores.hybrid_retriever.retrieve(
                query=q.question,
                top_k=config.top_k_retrieve,
                document_id=q.document_id or None,
                strategy="weighted",
                weights={"dense": 0.5, "bm25": 0.3, "visual": 0.2},
                include_dense=True,
                include_bm25=True,
                include_visual=True,
            ).results
            rerank_res = stores.reranker.rerank(
                query=q.question,
                candidates=candidates,
                top_k=top_k,
            )
            return rerank_res.results

        return config, retrieve_fn

    raise ValueError(f"Unknown baseline name '{baseline_name}'. Supported: {BASELINE_NAMES}")


def run_baseline_evaluation(
    stores: EvaluationStores,
    dataset: Sequence[GoldQuery] | GoldBenchmark,
    baseline_name: str,
    default_k: int = 5,
    k_values: Sequence[int] = (1, 3, 5, 10),
    thresholds: EvaluationThresholds | None = None,
) -> ExperimentResult:
    """Run retrieval evaluation for a specific named baseline."""
    config, ret_fn = get_retrieval_function_for_baseline(
        stores=stores, baseline_name=baseline_name, top_k=default_k
    )
    evaluator = RetrievalEvaluator(
        k_values=k_values, default_k=default_k, thresholds=thresholds
    )
    return evaluator.evaluate_queries(
        dataset=dataset,
        retrieval_fn=ret_fn,
        pipeline_config=config,
        dataset_id="gold-evaluation-v1",
        experiment_name=f"Baseline: {baseline_name}",
    )


def run_all_baselines(
    stores: EvaluationStores,
    dataset: Sequence[GoldQuery] | GoldBenchmark,
    default_k: int = 5,
    k_values: Sequence[int] = (1, 3, 5, 10),
    thresholds: EvaluationThresholds | None = None,
) -> dict[str, ExperimentResult]:
    """Execute evaluation across all standard named baselines."""
    results: dict[str, ExperimentResult] = {}
    for name in BASELINE_NAMES:
        results[name] = run_baseline_evaluation(
            stores=stores,
            dataset=dataset,
            baseline_name=name,
            default_k=default_k,
            k_values=k_values,
            thresholds=thresholds,
        )
    return results
