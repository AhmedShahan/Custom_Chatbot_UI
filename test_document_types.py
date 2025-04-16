#!/usr/bin/env python3
"""
Test script for the RAG system with different document types.
This script tests the ingestion and querying of PDF, PPT/PPTX, and DOC/DOCX files.
"""

import os
import sys
from rag_system import RAGSystem

def test_document(document_path, question="Tell me about this document"):
    """Test a document with the RAG system."""
    print(f"\n{'='*50}")
    print(f"Testing document: {document_path}")
    print(f"{'='*50}")
    
    # Get the file extension
    file_extension = os.path.splitext(document_path)[1].lower().lstrip('.')
    
    # Create RAG system
    rag = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
    
    # Process document based on file extension
    if file_extension == "pdf":
        rag.ingest_pdf(document_path)
    elif file_extension in ["ppt", "pptx"]:
        rag.ingest_ppt(document_path)
    elif file_extension in ["doc", "docx"]:
        rag.ingest_doc(document_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return
    
    # Test with hybrid approach
    print("\nQuery result (hybrid approach):")
    answer = rag.query(question)
    print(answer)
    
    return answer

def main():
    # Check if document paths are provided
    if len(sys.argv) < 2:
        print("Usage: python test_document_types.py <document_path1> [document_path2] ...")
        print("Example: python test_document_types.py sample.pdf sample.docx sample.pptx")
        return
    
    # Test each document
    for doc_path in sys.argv[1:]:
        if not os.path.exists(doc_path):
            print(f"Error: Document not found: {doc_path}")
            continue
        
        test_document(doc_path)

if __name__ == "__main__":
    main() 