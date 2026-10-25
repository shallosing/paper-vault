import json
import logging
import os
import sqlite3
import time
import codecs
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request as FastAPIRequest, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .evaluation import evaluate_retriever
from .retrieval import (
    chunk_text_with_offsets,
    first_stage_score,
    query_terms,
    rerank_candidates,
    select_first_stage,
)
from .schemas import (
    AskResponse,
    Document,
    DocumentContent,
    EvaluationRequest,
    EvaluationResponse,
    RetrievedChunk,
    SearchResult,
)

app = FastAPI(title="Paper Vault API", description="实验室论文资料管理、检索评估与问答助手")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE_PATH = BASE_DIR / "data" / "paper_vault.db"
LOG_DIR = BASE_DIR / "logs"
ALLOWED_SUFFIXES = {".txt", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

for directory in (UPLOAD_DIR, DATABASE_PATH.parent, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("paper_vault")
logger.setLevel(logging.INFO)
logger.propagate = False


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s request_id=%(request_id)s %(message)s")
    file_handler = RotatingFileHandler(
        LOG_DIR / "paper_vault.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
for handler in logger.handlers:
    handler.addFilter(RequestIdFilter())


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """
        )
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(chunks)")}
        if "start_char" not in existing:
            connection.execute("ALTER TABLE chunks ADD COLUMN start_char INTEGER")
        if "end_char" not in existing:
            connection.execute("ALTER TABLE chunks ADD COLUMN end_char INTEGER")


def find_saved_file(document_id: str) -> Path | None:
    files = list(UPLOAD_DIR.glob(f"{document_id}.*"))
    return files[0] if files else None


def read_uploaded_text(saved_file: Path) -> str:
    content = saved_file.read_bytes()
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return content.decode("utf-16")
    if content.startswith(codecs.BOM_UTF8):
        return content.decode("utf-8-sig")
    for encoding in ("utf-8", "gbk"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="无法识别文件编码，请使用 UTF-8、GBK 或 UTF-16 文本")


def index_document(document_id: str, content: str) -> int:
    chunks = chunk_text_with_offsets(content)
    with get_connection() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.executemany(
            """
            INSERT INTO chunks
                (id, document_id, chunk_index, content, start_char, end_char)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(str(uuid4()), document_id, index, chunk, start, end)
             for index, (start, end, chunk) in enumerate(chunks)],
        )
    return len(chunks)


def ensure_document_is_indexed(document_id: str) -> None:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT start_char FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
    if rows and all(row["start_char"] is not None for row in rows):
        return
    saved_file = find_saved_file(document_id)
    if saved_file is None:
        logger.warning("document file missing while indexing document_id=%s", document_id)
        return
    index_document(document_id, read_uploaded_text(saved_file))


def retrieve(question: str, top_k: int, use_reranker: bool = True) -> list[RetrievedChunk]:
    terms = query_terms(question)
    if not terms:
        raise HTTPException(status_code=400, detail="问题中需要包含可检索的关键词")

    with get_connection() as connection:
        document_ids = connection.execute("SELECT id FROM documents").fetchall()
    for document in document_ids:
        ensure_document_is_indexed(document["id"])

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT documents.id AS source_id, documents.filename,
                   chunks.chunk_index, chunks.content,
                   chunks.start_char, chunks.end_char
            FROM chunks JOIN documents ON chunks.document_id = documents.id
            """
        ).fetchall()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        score = first_stage_score(row["content"], terms)
        if score > 0:
            candidates.append({
                "source_id": row["source_id"],
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "content": row["content"],
                "first_stage_score": score,
            })

    first_stage = sorted(
        candidates,
        key=lambda item: (item["first_stage_score"], -item["chunk_index"]),
        reverse=True,
    )[:max(top_k * 10, 30)]
    ranked = (
        rerank_candidates(question, first_stage, top_k)
        if use_reranker
        else select_first_stage(first_stage, top_k)
    )
    return [RetrievedChunk.model_validate(item) for item in ranked]


def build_retrieval_only_answer(citations: list[RetrievedChunk]) -> str:
    if not citations:
        return "资料库中没有找到与该问题直接相关的内容。"
    excerpts = [f"[{item.citation_id}] {item.content}" for item in citations]
    return "当前为检索模式，以下片段是回答依据：\n\n" + "\n\n".join(excerpts)


def generate_with_llm(question: str, citations: list[RetrievedChunk]) -> str | None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or not citations:
        return None
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    context = "\n\n".join(
        f"[{item.citation_id}] 来源：{item.filename}，块号：{item.chunk_index}\n{item.content}"
        for item in citations
    )
    prompt = (
        "你是实验室论文问答助手。只能依据提供的片段回答；资料不足时明确说明。"
        "每个事实性结论都要标注来源编号，例如 [1]。不得编造来源编号。\n\n"
        f"资料片段：\n{context}\n\n问题：{question}"
    )
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "回答准确、简洁、引用可核查。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except (HTTPError, URLError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        logger.exception("llm request failed")
        raise HTTPException(status_code=502, detail="大模型调用失败，请检查配置或稍后重试") from error


init_db()


@app.middleware("http")
async def request_logging(request: FastAPIRequest, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request failed method=%s path=%s",
            request.method, request.url.path,
            extra={"request_id": request_id},
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request completed method=%s path=%s status=%s elapsed_ms=%.1f",
        request.method, request.url.path, response.status_code, elapsed_ms,
        extra={"request_id": request_id},
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[Document])
def list_documents():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, filename, size FROM documents ORDER BY rowid DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/documents", response_model=Document, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="目前只支持 .txt 和 .md 文件")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件不能超过 10 MB")

    document_id = str(uuid4())
    saved_path = UPLOAD_DIR / f"{document_id}{suffix}"
    saved_path.write_bytes(content)
    try:
        text = read_uploaded_text(saved_path)
        document = {"id": document_id, "filename": file.filename, "size": len(content)}
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO documents (id, filename, size) VALUES (:id, :filename, :size)",
                document,
            )
        index_document(document_id, text)
    except Exception:
        saved_path.unlink(missing_ok=True)
        with get_connection() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        logger.exception("document upload/index failed document_id=%s", document_id)
        raise
    return document


@app.get("/documents/{document_id}/content", response_model=DocumentContent)
def get_document_content(document_id: str):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, filename, size FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    saved_file = find_saved_file(document_id)
    if saved_file is None:
        raise HTTPException(status_code=404, detail="找不到对应的原始文件")
    return {**dict(row), "content": read_uploaded_text(saved_file)}


@app.get("/search", response_model=list[SearchResult])
def search_documents(q: str = Query(..., min_length=1, description="需要搜索的关键词")):
    keyword = q.strip().lower()
    if not keyword:
        raise HTTPException(status_code=400, detail="关键词不能为空")
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, filename, size FROM documents ORDER BY rowid DESC"
        ).fetchall()
    results = []
    for row in rows:
        saved_file = find_saved_file(row["id"])
        if saved_file is None:
            continue
        content = read_uploaded_text(saved_file)
        lowered = content.lower()
        position = lowered.find(keyword)
        if position == -1:
            continue
        start = max(0, position - 40)
        end = min(len(content), position + len(keyword) + 80)
        excerpt = content[start:end].replace("\n", " ")
        if start:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt += "..."
        results.append({**dict(row), "excerpt": excerpt, "occurrences": lowered.count(keyword)})
    return results


@app.get("/retrieve", response_model=list[RetrievedChunk])
def retrieve_documents(
    q: str = Query(..., min_length=1, description="检索问题"),
    top_k: int = Query(3, ge=1, le=10),
):
    return retrieve(q, top_k)


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_search(payload: EvaluationRequest):
    cases = [(case.question, case.relevant_document_ids) for case in payload.cases]
    return evaluate_retriever(
        cases,
        payload.k,
        lambda question, k: retrieve(question, k, payload.use_reranker),
        payload.use_reranker,
    )


@app.get("/ask", response_model=AskResponse)
def ask_question(
    q: str = Query(..., min_length=1, description="要向资料库提问的问题"),
    top_k: int = Query(3, ge=1, le=10),
):
    citations = retrieve(q, top_k)
    llm_answer = generate_with_llm(q, citations)
    if llm_answer is not None:
        answer, mode = llm_answer, "llm"
    else:
        answer, mode = build_retrieval_only_answer(citations), "retrieval_only"
    return AskResponse(question=q, answer=answer, mode=mode, citations=citations)


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="文件不存在")
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    saved_file = find_saved_file(document_id)
    if saved_file is not None:
        saved_file.unlink()
    return None
