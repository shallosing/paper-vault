import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Paper Vault API")

UPLOAD_DIR = Path("uploads")
DATABASE_PATH = Path("data/paper_vault.db")

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


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL
            )
            """
        )


def find_saved_file(document_id: str) -> Path | None:
    saved_files = list(UPLOAD_DIR.glob(f"{document_id}.*"))

    if not saved_files:
        return None

    return saved_files[0]


init_db()


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

    allowed_suffixes = {".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="目前只支持 .txt 和 .md 文件",
        )

    content = await file.read()
    document_id = str(uuid4())

    saved_path = UPLOAD_DIR / f"{document_id}{suffix}"
    saved_path.write_bytes(content)

    document = {
        "id": document_id,
        "filename": file.filename,
        "size": len(content),
    }

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (id, filename, size)
            VALUES (:id, :filename, :size)
            """,
            document,
        )

    return document


@app.get(
    "/documents/{document_id}/content",
    response_model=DocumentContent,
)
def get_document_content(document_id: str):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, size
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="文件不存在")

    saved_file = find_saved_file(document_id)

    if saved_file is None:
        raise HTTPException(status_code=404, detail="找不到对应的原始文件")

    try:
        content = saved_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件不是 UTF-8 文本")

    return {
        "id": row["id"],
        "filename": row["filename"],
        "size": row["size"],
        "content": content,
    }


@app.get("/search", response_model=list[SearchResult])
def search_documents(
    q: str = Query(..., min_length=1, description="需要搜索的关键词")
):
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

        try:
            content = saved_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

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
            excerpt = excerpt + "..."

        results.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "size": row["size"],
                "excerpt": excerpt,
                "occurrences": content_lower.count(keyword),
            }
        )

    return results


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="文件不存在")

        connection.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,),
        )

    saved_file = find_saved_file(document_id)

    if saved_file is not None:
        saved_file.unlink()

    return None