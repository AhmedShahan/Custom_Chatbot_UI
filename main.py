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

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_system
    
    # Save the uploaded file
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Initialize RAG system (with default settings for initial load)
    rag_system = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
    rag_system.ingest_pdf(file_path)
    
    return {"message": "PDF uploaded and processed successfully"}

@app.post("/ask")
async def ask_question(
    question: str = Query(...),
    model_type: str = Query(...),
    model_name: str = Query(...)
):
    global rag_system
    
    if rag_system is None:
        return {"error": "Please upload a PDF first"}
    
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