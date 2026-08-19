"""
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import rag_pipeline

app = FastAPI(
    title="Endometriosis Guideline Assistant",
    description="RAG API for NICE Endometriosis Guidelines",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Endometriosis Guideline Assistant is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, citation = rag_pipeline(request.question)
        return {"answer": answer, "sources": citation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
