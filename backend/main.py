"""
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from typing import Optional
from rag import INDEX_READY, STARTUP_ERROR, collection, rag_pipeline

app = FastAPI(
    title="Endometriosis Guideline Assistant",
    description="RAG API for NICE Endometriosis Guidelines",
    version="1.0.0",
)
@app.get("/")
def root():
    return {
        "message": "Endometriosis Guideline Assistant is running",
        "chunk_count": collection.count() if collection is not None else 0,
        "index_ready": INDEX_READY,
        "startup_error": STARTUP_ERROR,
    }


@app.post("/chat")
async def chat_endpoint(
    question: str = Form(""),
    pdf_file: Optional[UploadFile] = File(None),
):
    try:
        question = question.strip()
        pdf_bytes = None
        pdf_name = None

        if pdf_file is not None:
            if pdf_file.content_type not in (None, "application/pdf"):
                raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
            pdf_bytes = await pdf_file.read()
            pdf_name = pdf_file.filename
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

        answer, prompt, results = rag_pipeline(
            query=question,
            uploaded_pdf_bytes=pdf_bytes,
            uploaded_pdf_name=pdf_name,
        )

        sources = []
        for res in results:
            section_name = res.get("section_title") or res.get("section_name") or res.get("section", "Unknown")
            section_number = res.get("section_number", "Unknown")
            sources.append({
                "text": res.get("text", ""),
                "guideline": res.get("guideline", "Unknown"),
                "publisher": res.get("publisher", "Unknown"),
                "section": res.get("section", "Unknown"),
                "section_name": section_name,
                "section_title": section_name,
                "section_number": section_number,
                "page": res.get("page", "Unknown"),
                "chunk_id": res.get("chunk_id", "Unknown"),
                "source": res.get("source", "Unknown"),
                "source_type": res.get("source_type", "guideline"),
                "rerank_score": res.get("rerank_score", 0.0),
            })

        top_score = results[0].get("rerank_score", 0.0) if results else -2.0
        confidence = max(0.0, min(1.0, (top_score + 2) / 4)) if results else 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunk_count": collection.count() if collection is not None else 0,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
