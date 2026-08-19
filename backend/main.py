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
    # FIX: Use Form() for the text and File() for the document
    # Setting default values to "" and None makes them optional
    question: str = Form(""), 
    pdf_file: Optional[UploadFile] = File(None)
):
    try:
        pdf_bytes = None
        pdf_name = None
        
        # If a file was attached, read its bytes
        if pdf_file is not None:
            pdf_bytes = await pdf_file.read()
            pdf_name = pdf_file.filename

        # Pass the extracted data to your RAG pipeline
        answer, prompt, results = rag_pipeline(
            query=question,
            uploaded_pdf_bytes=pdf_bytes,
            uploaded_pdf_name=pdf_name
        )

        # Format the evidence to send back to Streamlit
        sources = []
        for res in results:
            sources.append({
                "text": res.get("text", ""),
                "guideline": res.get("guideline", "Unknown"),
                "section": res.get("section", "Unknown"),
                "page": res.get("page", "Unknown"),
                "chunk_id": res.get("chunk_id", "Unknown"),
                "rerank_score": res.get("rerank_score", 0.0)
            })

        # Calculate a mock confidence score based on the top rerank score
        top_score = results[0].get("rerank_score", 0.0) if results else -2.0
        confidence = max(0.0, min(1.0, (top_score + 2) / 4)) if results else 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "confidence": 0.0}
