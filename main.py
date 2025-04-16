from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from rag_system import RAGSystem
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    global rag_system
    
    # Get file extension
    file_extension = file.filename.split(".")[-1].lower()
    
    # Determine the appropriate directory based on file type
    if file_extension == "pdf":
        save_dir = "uploads/pdf"
    elif file_extension in ["ppt", "pptx"]:
        save_dir = "uploads/ppt"
    elif file_extension in ["doc", "docx"]:
        save_dir = "uploads/doc"
    else:
        return {"error": f"Unsupported file format: {file_extension}"}
    
    # Save the uploaded file
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Initialize RAG system (with default settings for initial load)
    rag_system = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
    
    # Process the file based on its extension
    if file_extension == "pdf":
        rag_system.ingest_pdf(file_path)
    elif file_extension in ["ppt", "pptx"]:
        rag_system.ingest_ppt(file_path)
    elif file_extension in ["doc", "docx"]:
        rag_system.ingest_doc(file_path)
    else:
        return {"error": f"Unsupported file format: {file_extension}"}
    
    return {"message": f"Document uploaded and processed successfully"}

# Keep the old endpoint for backward compatibility
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    return await upload_document(file)

@app.post("/ask")
async def ask_question(
    question: str = Query(...),
    model_type: str = Query(...),
    model_name: str = Query(...)
):
    global rag_system
    
    if rag_system is None:
        return {"error": "Please upload a document first"}
    
    try:
        # Configure RAG system based on selection
        if model_type == "LLM":
            rag_system.use_llm = True
            rag_system.model_name = model_name
            rag_system.method = "hybrid"
        else:
            rag_system.use_llm = False
            rag_system.method = model_name
        
        # Get answer from RAG system
        answer = rag_system.query(question)
        
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Error processing question: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)