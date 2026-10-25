from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    filename: str
    size: int


class DocumentContent(Document):
    content: str


class SearchResult(Document):
    excerpt: str
    occurrences: int


class RetrievedChunk(BaseModel):
    citation_id: int
    source_id: str
    filename: str
    chunk_index: int
    source_ref: str
    start_char: int | None = None
    end_char: int | None = None
    content: str
    first_stage_score: float
    rerank_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["llm", "retrieval_only"]
    citations: list[RetrievedChunk]


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(min_length=1)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)
    use_reranker: bool = True


class EvaluationCaseResult(BaseModel):
    question: str
    relevant_document_ids: list[str]
    retrieved_document_ids: list[str]
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


class EvaluationResponse(BaseModel):
    k: int
    use_reranker: bool
    case_count: int
    mean_hit_at_k: float
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    results: list[EvaluationCaseResult]
