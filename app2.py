# app.py
import os
import tempfile
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import shutil
from contextlib import asynccontextmanager

# Import the RAG system without LLM
from rag_without_llm import RAGSystemWithoutLLM

# Store sessions in memory for simplicity
# In production, use a database or cache system
sessions = {}

# Define models for API
class QueryRequest(BaseModel):
    session_id: str
    question: str
    method: str = "extractive"  # Default method

class QueryResponse(BaseModel):
    responses: Dict[str, str]

class SessionInfo(BaseModel):
    methods: List[str]
    document_name: str
    status: str

def get_methods():
    # List of available response generation methods
    return ["retrieval", "template", "extractive", "rule", "extraction", "tfidf", "keyword"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Cleanup logic for sessions when app shuts down
    yield
    
    # Cleanup temp files
    for session_id in sessions:
        session_path = os.path.join("data", session_id)
        if os.path.exists(session_path):
            shutil.rmtree(session_path)

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    methods: List[str] = Form(...)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    session_dir = os.path.join("data", session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Save the uploaded file
    temp_file_path = os.path.join(session_dir, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Validate methods
    available_methods = get_methods()
    valid_methods = [method for method in methods if method in available_methods]
    
    # If no valid methods, use default
    if not valid_methods:
        valid_methods = ["extractive"]
    
    # Initialize session information
    sessions[session_id] = {
        "file_path": temp_file_path,
        "methods": valid_methods,
        "document_name": file.filename,
        "status": "processing",
        "rag_instance": None
    }
    
    # Process document asynchronously
    asyncio.create_task(process_document(session_id, temp_file_path))
    
    return {"session_id": session_id, "status": "processing"}

async def process_document(session_id: str, file_path: str):
    try:
        # Create a single RAG instance (no need for multiple models since we're not using LLMs)
        rag = RAGSystemWithoutLLM()
        rag.ingest_pdf(file_path)
        
        # Save the processed data
        model_path = os.path.join("data", session_id, "vector_store")
        rag.save(model_path)
        
        # Store the RAG instance
        sessions[session_id]["rag_instance"] = rag
        sessions[session_id]["status"] = "ready"
    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "status": session["status"],
        "document_name": session["document_name"],
        "methods": session["methods"],
        "error": session.get("error", None)
    }

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"Session is not ready yet. Current status: {session['status']}")
    
    responses = {}
    
    # Get the RAG instance
    rag_instance = session["rag_instance"]
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not found for this session")
    
    # Query using all methods in the session or just the requested method
    methods_to_query = []
    if request.method == "all":
        methods_to_query = session["methods"]
    elif request.method in get_methods():
        methods_to_query = [request.method]
    else:
        # Default to extractive if invalid method
        methods_to_query = ["extractive"]
    
    # Get responses for each method
    for method in methods_to_query:
        try:
            response = rag_instance.query(request.question, method=method)
            
            # If response is too generic or empty, assume no match was found
            lower_response = response.lower()
            if (not response or 
                "no relevant information" in lower_response or
                "no information found" in lower_response):
                response = "This information is not available in the document."
                
            responses[method] = response
        except Exception as e:
            responses[method] = f"Error using method {method}: {str(e)}"
    
    return {"responses": responses}

@app.get("/methods")
async def list_methods():
    return {"methods": get_methods()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)