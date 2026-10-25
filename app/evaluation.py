from collections.abc import Callable

from .schemas import EvaluationCaseResult, EvaluationResponse, RetrievedChunk


def evaluate_retriever(
    cases: list[tuple[str, list[str]]],
    k: int,
    retrieve: Callable[[str, int], list[RetrievedChunk]],
    use_reranker: bool = True,
) -> EvaluationResponse:
    """对人工标注的相关文档 ID 计算常用排序指标。"""
    results: list[EvaluationCaseResult] = []

    for question, relevant_ids in cases:
        relevant = set(relevant_ids)
        chunks = retrieve(question, k)
        ranked_documents = list(dict.fromkeys(chunk.source_id for chunk in chunks))
        top_documents = ranked_documents[:k]
        relevant_hits = [doc_id for doc_id in top_documents if doc_id in relevant]
        first_relevant_rank = next(
            (rank for rank, doc_id in enumerate(ranked_documents, start=1) if doc_id in relevant),
            None,
        )

        results.append(
            EvaluationCaseResult(
                question=question,
                relevant_document_ids=sorted(relevant),
                retrieved_document_ids=top_documents,
                hit_at_k=float(bool(relevant_hits)),
                recall_at_k=len(relevant_hits) / len(relevant),
                precision_at_k=len(relevant_hits) / k,
                reciprocal_rank=1 / first_relevant_rank if first_relevant_rank else 0.0,
            )
        )

    count = len(results)
    return EvaluationResponse(
        k=k,
        case_count=count,
        mean_hit_at_k=sum(item.hit_at_k for item in results) / count,
        mean_recall_at_k=sum(item.recall_at_k for item in results) / count,
        mean_precision_at_k=sum(item.precision_at_k for item in results) / count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in results) / count,
        use_reranker=use_reranker,
        results=results,
    )
