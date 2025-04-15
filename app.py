from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from rag_system import RAGSystem  # আপনার সিস্টেম থেকে আমদানি

app = FastAPI(title="RAG Chatbot API")

# CORS সেটিংস (Streamlit এর সাথে যোগাযোগের জন্য)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# সাময়িক PDF সংরক্ষণের জন্য ফোল্ডার
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# RAG সিস্টেম সংরক্ষণের জন্য ফোল্ডার
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# সক্রিয় RAG সিস্টেম
active_rag = None

@app.post("/upload_and_train")
async def upload_and_train(
    file: UploadFile = File(...),
    model_name: str = Form("gemma3:latest"),
    use_llm: bool = Form(True),
    method: str = Form("hybrid")
):
    """PDF আপলোড এবং RAG সিস্টেম প্রশিক্ষণ"""
    try:
        # সাময়িক ফাইল সংরক্ষণ
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # RAG সিস্টেম তৈরি ও প্রশিক্ষণ
        global active_rag
        active_rag = RAGSystem(model_name=model_name, use_llm=use_llm, method=method)
        active_rag.ingest_pdf(file_path)
        
        # মডেল সংরক্ষণ
        model_path_prefix = os.path.join(MODEL_DIR, f"{os.path.splitext(file.filename)[0]}")
        active_rag.save(model_path_prefix)
        
        return {"status": "success", "message": "PDF আপলোড এবং RAG সিস্টেম প্রশিক্ষণ সম্পন্ন হয়েছে"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"সার্ভার ত্রুটি: {str(e)}")

@app.post("/load_model")
async def load_model(
    filename: str = Form(...),
    model_name: str = Form("gemma3:latest"),
    use_llm: bool = Form(True),
    method: str = Form("hybrid")
):
    """সংরক্ষিত RAG মডেল লোড করা"""
    try:
        model_path_prefix = os.path.join(MODEL_DIR, f"{filename}")
        
        global active_rag
        active_rag = RAGSystem.load(
            model_path_prefix, 
            model_name=model_name,
            use_llm=use_llm,
            method=method
        )
        
        return {"status": "success", "message": f"মডেল '{filename}' সফলভাবে লোড হয়েছে"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"মডেল লোড ত্রুটি: {str(e)}")

@app.post("/query")
async def query(
    question: str = Form(...),
    force_llm: bool = Form(False),
    k: int = Form(5)
):
    """প্রশ্ন জিজ্ঞাসা করা"""
    try:
        if active_rag is None:
            raise HTTPException(status_code=400, detail="কোন সক্রিয় RAG মডেল লোড করা হয়নি")
        
        result = active_rag.query(question, k=k, force_llm=force_llm)
        return {"status": "success", "answer": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"প্রশ্ন উত্তর দেওয়ার সময় ত্রুটি: {str(e)}")

@app.get("/saved_models")
async def list_models():
    """সংরক্ষিত মডেলগুলি তালিকাভুক্ত করা"""
    try:
        # ডিরেক্টরিতে উপস্থিত ফাইলগুলি খুঁজে বের করা
        model_files = {}
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith("_documents.json"):
                model_name = filename.replace("_documents.json", "")
                model_files[model_name] = os.path.getmtime(os.path.join(MODEL_DIR, filename))
        
        # তালিকা সৃষ্টি
        models = [
            {"name": name, "created": timestamp}
            for name, timestamp in model_files.items()
        ]
        
        # সময় অনুযায়ী সাজানো
        models.sort(key=lambda x: x["created"], reverse=True)
        
        return {"status": "success", "models": models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"মডেল তালিকা আনতে ত্রুটি: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)