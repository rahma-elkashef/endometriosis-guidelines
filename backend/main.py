"""
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.rag import INDEX_READY, STARTUP_ERROR, collection, rag_pipeline

app = FastAPI(
    title="Endometriosis Guideline Assistant",
    description="RAG API for NICE Endometriosis Guidelines",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Endometriosis Guideline Assistant is running",
        "chunk_count": collection.count() if collection is not None else 0,
        "index_ready": INDEX_READY,
        "startup_error": STARTUP_ERROR,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not INDEX_READY:
        raise HTTPException(
            status_code=503,
            detail=STARTUP_ERROR or "RAG index is not available. Run the indexing pipeline first.",
        )

    try:
        answer, prompt, results = rag_pipeline(request.question)

        confidence = 0.0
        if results:
            top_result = results[0]
            rerank_score = float(top_result.get("rerank_score", 0.0))
            confidence = max(0.0, min(1.0, (rerank_score + 5.0) / 10.0))

            chroma_distance = top_result.get("chroma_distance")
            if chroma_distance is not None:
                distance_confidence = max(0.0, min(1.0, 1.0 - float(chroma_distance)))
                confidence = max(confidence, distance_confidence)

        return {
            "answer": answer,
            "sources": results,
            "prompt": prompt,
            "confidence": confidence,
            "chunk_count": collection.count() if collection is not None else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
