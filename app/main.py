import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Paper Vault API", description="实验室论文资料管理与检索问答助手 v1")

UPLOAD_DIR = Path("uploads")
DATABASE_PATH = Path("data/paper_vault.db")
ALLOWED_SUFFIXES = {".txt", ".md"}
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

UPLOAD_DIR.mkdir(exist_ok=True)
DATABASE_PATH.parent.mkdir(exist_ok=True)


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
    source_id: str
    filename: str
    chunk_index: int
    content: str
    score: int


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["llm", "retrieval_only"]
    citations: list[RetrievedChunk]


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
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id);
            """
        )


def find_saved_file(document_id: str) -> Path | None:
    saved_files = list(UPLOAD_DIR.glob(f"{document_id}.*"))
    return saved_files[0] if saved_files else None


# def read_uploaded_text(saved_file: Path) -> str:
#     """支持 PowerShell 常见的 UTF-8 与 UTF-16 文本编码。"""
#     for encoding in ("utf-8-sig", "utf-16"):
#         try:
#             return saved_file.read_text(encoding=encoding)
#         except UnicodeDecodeError:
#             continue
#     raise HTTPException(status_code=400, detail="暂不支持该文件编码，请使用 UTF-8 或 UTF-16 文本")

def read_uploaded_text(saved_file: Path) -> str | None:
    """支持 UTF-8(BOM)、GBK、GB2312、UTF-16。解码失败返回None"""
    for encoding in ("utf-8-sig", "gbk", "gb2312", "utf-16"):
        try:
            return saved_file.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return None


def chunk_text(text: str) -> list[str]:
    """把长文本切成带重叠的小块，避免检索时丢失上下文。"""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = max(
                normalized.rfind(mark, start + CHUNK_SIZE // 2, end)
                for mark in "。！？；.!?;"
            )
            if boundary != -1:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def index_document(document_id: str, content: str) -> int:
    chunks = chunk_text(content)
    with get_connection() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.executemany(
            """
            INSERT INTO chunks (id, document_id, chunk_index, content)
            VALUES (?, ?, ?, ?)
            """,
            [(str(uuid4()), document_id, index, chunk) for index, chunk in enumerate(chunks)],
        )
    return len(chunks)


# def ensure_document_is_indexed(document_id: str) -> None:
#     """兼容升级前上传的旧文档：首次搜索时自动补建索引。"""
#     with get_connection() as connection:
#         row = connection.execute(
#             "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1", (document_id,)
#         ).fetchone()
#     if row is not None:
#         return
#     saved_file = find_saved_file(document_id)
#     if saved_file is not None:
#         index_document(document_id, read_uploaded_text(saved_file))
def ensure_document_is_indexed(document_id: str) -> None:
    """首次检索自动建索引，编码无法解析的文档直接跳过"""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1", (document_id,)
        ).fetchone()
    if row is not None:
        return
    saved_file = find_saved_file(document_id)
    if saved_file is not None:
        content = read_uploaded_text(saved_file)
        if content is None:
            return
        index_document(document_id, content)


def query_terms(query: str) -> set[str]:
    """提取英文单词与中文二元词，提供零依赖的基础检索能力。"""
    query = query.lower()
    english_terms = re.findall(r"[a-z0-9_]+", query)
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", query)
    chinese_bigrams = [
        sequence[index : index + 2]
        for sequence in chinese_sequences
        for index in range(len(sequence) - 1)
    ]
    return set(english_terms + chinese_bigrams + chinese_sequences)


def retrieve(question: str, top_k: int) -> list[RetrievedChunk]:
    terms = query_terms(question)
    if not terms:
        raise HTTPException(status_code=400, detail="问题中需要包含可检索的关键词")

    with get_connection() as connection:
        documents = connection.execute("SELECT id FROM documents").fetchall()
    for document in documents:
        ensure_document_is_indexed(document["id"])

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT documents.id AS source_id, documents.filename,
                   chunks.chunk_index, chunks.content
            FROM chunks JOIN documents ON chunks.document_id = documents.id
            """
        ).fetchall()

    candidates: list[RetrievedChunk] = []
    for row in rows:
        content_lower = row["content"].lower()
        score = sum(content_lower.count(term) for term in terms)
        if score:
            candidates.append(
                RetrievedChunk(
                    source_id=row["source_id"], filename=row["filename"],
                    chunk_index=row["chunk_index"], content=row["content"], score=score,
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


def build_retrieval_only_answer(citations: list[RetrievedChunk]) -> str:
    if not citations:
        return "资料库中没有找到与该问题直接相关的内容。"
    excerpts = [f"[{index}] {citation.content}" for index, citation in enumerate(citations, 1)]
    return "未配置大模型 API，以下是最相关的资料片段：\n\n" + "\n\n".join(excerpts)


def generate_with_llm(question: str, citations: list[RetrievedChunk]) -> str | None:
    """可选：调用任何 OpenAI 兼容的 Chat Completions 接口。"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    context = "\n\n".join(
        f"[{index}] 文件：{citation.filename}\n{citation.content}"
        for index, citation in enumerate(citations, 1)
    )
    prompt = (
        "你是实验室论文问答助手。只能依据给出的资料回答；资料不足时请明确说明。"
        "每个关键结论后标注来源编号，如 [1]。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    payload = json.dumps(
        {"model": model, "messages": [
            {"role": "system", "content": "回答应简洁、准确、可追溯。"},
            {"role": "user", "content": prompt},
        ], "temperature": 0.2}
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except (HTTPError, URLError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"大模型调用失败：{error}")


init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[Document])
def list_documents():
    with get_connection() as connection:
        rows = connection.execute("SELECT id, filename, size FROM documents ORDER BY rowid DESC").fetchall()
    return [dict(row) for row in rows]


# @app.post("/documents", response_model=Document, status_code=201)
# async def upload_document(file: UploadFile = File(...)):
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="文件名不能为空")
#     suffix = Path(file.filename).suffix.lower()
#     if suffix not in ALLOWED_SUFFIXES:
#         raise HTTPException(status_code=400, detail="目前只支持 .txt 和 .md 文件")

#     content = await file.read()
#     document_id = str(uuid4())
#     saved_path = UPLOAD_DIR / f"{document_id}{suffix}"
#     saved_path.write_bytes(content)
#     document = {"id": document_id, "filename": file.filename, "size": len(content)}
#     with get_connection() as connection:
#         connection.execute("INSERT INTO documents (id, filename, size) VALUES (:id, :filename, :size)", document)
#     try:
#         index_document(document_id, read_uploaded_text(saved_path))
#     except HTTPException:
#         saved_path.unlink(missing_ok=True)
#         with get_connection() as connection:
#             connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
#         raise
#     return document
@app.post("/documents", response_model=Document, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="目前只支持 .txt 和 .md 文件")
    content = await file.read()
    document_id = str(uuid4())
    saved_path = UPLOAD_DIR / f"{document_id}{suffix}"
    saved_path.write_bytes(content)
    document = {"id": document_id, "filename": file.filename, "size": len(content)}
    with get_connection() as connection:
        connection.execute("INSERT INTO documents (id, filename, size) VALUES (:id, :filename, :size)", document)
    try:
        text = read_uploaded_text(saved_path)
        if text is None:
            raise HTTPException(status_code=400, detail="暂不支持该文件编码，请使用 UTF-8 / GBK / UTF-16 文本")
        index_document(document_id, text)
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        with get_connection() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        raise
    return document



@app.get("/documents/{document_id}/content", response_model=DocumentContent)
def get_document_content(document_id: str):
    with get_connection() as connection:
        row = connection.execute("SELECT id, filename, size FROM documents WHERE id = ?", (document_id,)).fetchone()
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
        rows = connection.execute("SELECT id, filename, size FROM documents ORDER BY rowid DESC").fetchall()
    results = []
    for row in rows:
        saved_file = find_saved_file(row["id"])
        if saved_file is None:
            continue
        content = read_uploaded_text(saved_file)
        content_lower = content.lower()
        first_position = content_lower.find(keyword)
        if first_position == -1:
            continue
        start = max(0, first_position - 40)
        end = min(len(content), first_position + len(keyword) + 80)
        excerpt = content[start:end].replace("\n", " ")
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt += "..."
        results.append({**dict(row), "excerpt": excerpt, "occurrences": content_lower.count(keyword)})
    return results


@app.get("/retrieve", response_model=list[RetrievedChunk])
def retrieve_documents(
    q: str = Query(..., min_length=1, description="需要检索的问题"),
    top_k: int = Query(3, ge=1, le=10, description="返回最相关的文本块数量"),
):
    return retrieve(q, top_k)


@app.get("/ask", response_model=AskResponse)
def ask_question(
    q: str = Query(..., min_length=1, description="要向资料库提问的问题"),
    top_k: int = Query(3, ge=1, le=10, description="提供给回答器的资料块数量"),
):
    citations = retrieve(q, top_k)
    llm_answer = generate_with_llm(q, citations)
    if llm_answer is not None:
        return AskResponse(question=q, answer=llm_answer, mode="llm", citations=citations)
    return AskResponse(
        question=q, answer=build_retrieval_only_answer(citations),
        mode="retrieval_only", citations=citations,
    )


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    with get_connection() as connection:
        row = connection.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="文件不存在")
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    saved_file = find_saved_file(document_id)
    if saved_file is not None:
        saved_file.unlink()
    return None
