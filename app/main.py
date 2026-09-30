import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
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

    saved_files = list(UPLOAD_DIR.glob(f"{document_id}.*"))
    for saved_file in saved_files:
        saved_file.unlink()

    return None