from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from rag_system import RAGSystem
import os
import requests
from urllib.parse import urlparse
from website_scraper import WebsiteScraper

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
    elif file_extension in ["csv", "xls", "xlsx"]:
        save_dir = "uploads/spreadsheet"
        print(f"\n\n{'='*50}")
        print(f"Processing spreadsheet: {file.filename}")
        print(f"File extension detected: {file_extension}")
        print(f"{'='*50}")
    else:
        return {"error": f"Unsupported file format: {file_extension}"}
    
    # Save the uploaded file
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    print(f"\n\n{'='*50}")
    print(f"File saved to: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    print(f"{'='*50}")
    
    # Initialize RAG system (with default settings for initial load)
    rag_system = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
    
    # Process the file based on its extension
    success = False
    try:
        if file_extension == "pdf":
            rag_system.ingest_pdf(file_path)
            success = True
        elif file_extension in ["ppt", "pptx"]:
            # Use our improved PPT processing method that converts to PDF first
            success = rag_system.ingest_ppt(file_path)
        elif file_extension in ["doc", "docx"]:
            rag_system.ingest_doc(file_path)
            success = True
        elif file_extension in ["csv", "xls", "xlsx"]:
            print(f"Calling ingest_spreadsheet for file: {file_path}")
            success = rag_system.ingest_spreadsheet(file_path)
            print(f"ingest_spreadsheet returned: {success}")
        else:
            return {"error": f"Unsupported file format: {file_extension}"}
    except Exception as e:
        import traceback
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error processing document: {str(e)}"}
    
    # Check if any documents were successfully added
    doc_count = len(rag_system.vector_store.documents)
    print(f"Document count in vector store: {doc_count}")
    
    if success and doc_count > 0:
        print(f"Document processed successfully with {doc_count} sections")
        return {
            "message": f"Document uploaded and processed successfully", 
            "doc_count": doc_count
        }
    else:
        print("Document processing failed or no content extracted")
        return {"error": f"Document was saved but processing failed. Please try a different file or format."}

# Keep the old endpoint for backward compatibility
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    return await upload_document(file)

@app.post("/upload-spreadsheet")
async def upload_spreadsheet(file: UploadFile = File(...)):
    global rag_system
    
    # Get file extension
    file_extension = file.filename.split(".")[-1].lower()
    
    # Determine the appropriate directory based on file type
    if file_extension in ["csv", "xls", "xlsx"]:
        save_dir = "uploads/spreadsheet"
    else:
        return {"error": f"Unsupported file format: {file_extension}"}
    
    # Save the uploaded file
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    print(f"\n\n{'='*50}")
    print(f"Processing spreadsheet: {file.filename}")
    print(f"{'='*50}")
    
    # Initialize RAG system (with default settings for initial load)
    rag_system = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
    
    # Process the file based on its extension
    success = False
    try:
        if file_extension in ["csv", "xls", "xlsx"]:
            success = rag_system.ingest_spreadsheet(file_path)
        else:
            return {"error": f"Unsupported file format: {file_extension}"}
    except Exception as e:
        import traceback
        print(f"Error processing spreadsheet: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error processing spreadsheet: {str(e)}"}
    
    # Check if any documents were successfully added
    doc_count = len(rag_system.vector_store.documents)
    
    if success and doc_count > 0:
        print(f"Spreadsheet processed successfully with {doc_count} sections")
        return {
            "message": f"Spreadsheet uploaded and processed successfully", 
            "doc_count": doc_count
        }
    else:
        print("Spreadsheet processing failed or no content extracted")
        return {"error": f"Spreadsheet was saved but processing failed. Please try a different file or format."}

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

@app.get("/document-status")
async def document_status():
    """Get information about the currently loaded document"""
    global rag_system
    
    if rag_system is None:
        return {"status": "No document loaded"}
        
    doc_count = len(rag_system.vector_store.documents)
    
    # Get summary of the documents
    doc_summary = []
    if doc_count > 0:
        for i, (doc_id, doc) in enumerate(rag_system.vector_store.documents.items()):
            summary = {
                "id": doc_id[:8],  # Shortened ID
                "title": doc.title[:50] + "..." if len(doc.title) > 50 else doc.title,
                "text_length": len(doc.text),
                "keywords": doc.keywords[:5] if doc.keywords else []
            }
            doc_summary.append(summary)
            if i >= 9:  # Limit to 10 documents for the response
                break
    
    return {
        "status": "Document loaded",
        "document_count": doc_count,
        "documents": doc_summary
    }

@app.post("/process-website")
async def process_website(url: str = Query(...)):
    global rag_system
    
    # Validate URL
    try:
        # Check if URL is valid
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return {"error": "Invalid URL format"}
        
        # Check if website is accessible
        response = requests.head(url, timeout=10)
        if response.status_code >= 400:
            return {"error": f"Website returned error status: {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Failed to access website: {str(e)}"}
    
    try:
        print(f"\n\n{'='*50}")
        print(f"Processing website: {url}")
        print(f"{'='*50}")
        
        # Create output directories if they don't exist
        os.makedirs("uploads/website", exist_ok=True)
        os.makedirs("uploads/pdf", exist_ok=True)
        
        # Create a unique filename based on the domain
        domain = urlparse(url).netloc
        safe_domain = ''.join(c if c.isalnum() else '_' for c in domain)
        output_filename = f"{safe_domain}.pdf"
        
        # Initialize scraper and process website
        scraper = WebsiteScraper(url, output_folder="uploads/website")
        scraper.scrape_website()
        pdf_path = scraper.convert_to_pdf(output_filename)
        
        if not pdf_path or not os.path.exists(pdf_path):
            return {"error": "Failed to generate PDF from website content"}
        
        # Initialize RAG system (with default settings for initial load)
        rag_system = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
        
        # Process the generated PDF
        rag_system.ingest_pdf(pdf_path)
        
        # Check if any documents were successfully added
        doc_count = len(rag_system.vector_store.documents)
        
        if doc_count > 0:
            print(f"Website processed successfully with {doc_count} sections")
            return {
                "message": f"Website processed successfully", 
                "doc_count": doc_count
            }
        else:
            print("Website processing failed or no content extracted")
            return {"error": "Website was processed but no useful content was extracted"}
    except Exception as e:
        import traceback
        print(f"Error processing website: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error processing website: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)