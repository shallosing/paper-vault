import re
from collections import Counter
from typing import Any


CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def chunk_text_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """按字符切片并保留原文位置，供引用定位使用。"""
    if not text.strip():
        return []

    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(mark, start + CHUNK_SIZE // 2, end)
                for mark in "。！？；.!?;\n"
            )
            if boundary != -1:
                end = boundary + 1

        left = start
        right = end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            chunks.append((left, right, text[left:right]))

        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def query_terms(query: str) -> list[str]:
    """提取英文词和中文二元词；用 list 保留可预测的顺序。"""
    lowered = query.lower()
    english_terms = re.findall(r"[a-z0-9_]+", lowered)
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese_bigrams = [
        sequence[index : index + 2]
        for sequence in chinese_sequences
        for index in range(len(sequence) - 1)
    ]
    return list(dict.fromkeys(english_terms + chinese_bigrams + chinese_sequences))


def first_stage_score(content: str, terms: list[str]) -> float:
    lowered = content.lower()
    return float(sum(lowered.count(term) for term in terms))


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """用词覆盖率、词频和整句命中做轻量启发式重排（不是神经 cross-encoder）。"""
    terms = query_terms(question)
    question_phrase = re.sub(r"\s+", "", question.lower())

    for candidate in candidates:
        content = candidate["content"].lower()
        counts = Counter({term: content.count(term) for term in terms})
        covered = sum(count > 0 for count in counts.values())
        coverage = covered / max(len(terms), 1)
        frequency = sum(min(count, 3) for count in counts.values())
        phrase_bonus = 2.0 if question_phrase and question_phrase in re.sub(r"\s+", "", content) else 0.0
        candidate["rerank_score"] = round(coverage * 5.0 + frequency * 0.25 + phrase_bonus, 4)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["rerank_score"],
            candidate["first_stage_score"],
            -candidate["chunk_index"],
        ),
        reverse=True,
    )[:top_k]

    for citation_id, candidate in enumerate(ranked, start=1):
        candidate["citation_id"] = citation_id
        candidate["source_ref"] = (
            f"doc:{candidate['source_id']}#chunk={candidate['chunk_index']}"
        )
    return ranked


def select_first_stage(
    candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """直接按召回阶段分数排序，便于和重排结果做离线对照。"""
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["first_stage_score"],
            -candidate["chunk_index"],
        ),
        reverse=True,
    )[:top_k]
    for citation_id, candidate in enumerate(ranked, start=1):
        candidate["citation_id"] = citation_id
        candidate["rerank_score"] = candidate["first_stage_score"]
        candidate["source_ref"] = (
            f"doc:{candidate['source_id']}#chunk={candidate['chunk_index']}"
        )
    return ranked
