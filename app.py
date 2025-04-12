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

# Import the RAG system
# from rag_system import RAGSystem
from rag_system2 import RAGSystemWithoutLLM


# Store sessions in memory for simplicity
# In production, use a database or cache system
sessions = {}

# Define models for API
class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    responses: Dict[str, str]

class SessionInfo(BaseModel):
    models: List[str]
    document_name: str
    status: str

def get_models():
    # List of available models
    return ["gemma3:latest", "deepseek-r1:14b", "llama3.2:1b"]

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
async def upload_document(file: UploadFile = File(...), models: List[str] = Form(...)):
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
    
    # Initialize session information
    sessions[session_id] = {
        "file_path": temp_file_path,
        "models": models,
        "document_name": file.filename,
        "status": "processing",
        "rag_instances": {}
    }
    
    # Process document asynchronously
    asyncio.create_task(process_document(session_id, temp_file_path, models))
    
    return {"session_id": session_id, "status": "processing"}

async def process_document(session_id: str, file_path: str, models: List[str]):
    try:
        available_models = get_models()
        for model in models:
            if model in available_models:
                # Create RAG instance for each model
                rag = RAGSystemWithoutLLM(model_name=model)
                rag.ingest_pdf(file_path)
                
                # Save the processed data
                model_path = os.path.join("data", session_id, model.replace(":", "_"))
                rag.save(model_path)
                
                # Store the RAG instance
                sessions[session_id]["rag_instances"][model] = rag
        
        # Update session status
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
        "models": session["models"],
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
    
    # Query each model
    for model, rag_instance in session["rag_instances"].items():
        try:
            response = rag_instance.query(request.question)
            # If response is too generic or empty, assume no match was found
            lower_response = response.lower()
            if (not response or 
                "i don't have enough information" in lower_response or
                "cannot answer" in lower_response or
                "no relevant information" in lower_response):
                response = "This information is not available in the document."
            responses[model] = response
        except Exception as e:
            responses[model] = f"Error querying model {model}: {str(e)}"
    
    return {"responses": responses}

@app.get("/models")
async def list_models():
    return {"models": get_models()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)