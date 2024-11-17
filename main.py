from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from sales_ai_assistant import ai_agent
from rag_hybridsearch import (
    process_pdf,
    chunk_text,
    create_dense_embeddings,
    create_sparse_embeddings,
    upsert_to_pinecone,
    hybrid_query,
    generate_answer,
)
import os
import shutil

# Initialize FastAPI
app = FastAPI()

# Directory for uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Models for API
class QueryRequest(BaseModel):
    query: str
    alpha: float = 0.5
    top_k: int = 5


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file and process it for hybrid search.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process the PDF file
    text = process_pdf(file_path)
    chunks = chunk_text(text)
    dense_embeddings = create_dense_embeddings(chunks)
    sparse_embeddings = create_sparse_embeddings(text, chunks)
    upsert_to_pinecone(chunks, dense_embeddings, sparse_embeddings)

    return {"message": f"File {file.filename} uploaded and processed successfully."}


@app.post("/query/")
async def query_index(request: QueryRequest):
    """
    Endpoint to query the hybrid search index and get results.
    """
    matches = hybrid_query(request.query, alpha=request.alpha, top_k=request.top_k)
    context = [match["metadata"]["text"] for match in matches]
    answer = generate_answer(request.query, context)
    return {"query": request.query, "answer": answer, "matches": matches}


@app.post("/sales_ai/")
async def sales_query(question: str):
    """
    Endpoint to query the Sales AI Assistant.
    """
    try:
        answer = ai_agent(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint to check API status.
    """
    return {"message": "Sales Assistant and RAG Hybrid Search API are running!"}
