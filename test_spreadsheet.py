"""
Test script to verify spreadsheet conversion functionality
"""
import os
import sys
from rag_system import RAGSystem

def test_spreadsheet_conversion(file_path):
    """Test the spreadsheet conversion functionality"""
    print(f"Testing spreadsheet conversion for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False
    
    # Create RAG system
    rag = RAGSystem(model_name="deepseek-r1:14b", use_llm=True, method="hybrid")
    
    # Process the spreadsheet file
    success = rag.ingest_spreadsheet(file_path)
    
    # Check result
    if success:
        doc_count = len(rag.vector_store.documents)
        print(f"Successfully processed spreadsheet with {doc_count} sections")
        return True
    else:
        print("Failed to process spreadsheet")
        return False

if __name__ == "__main__":
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default test file
        file_path = "uploads/spreadsheet/test.xlsx"  # Update with your test file
        
        # Check if we have any xlsx files in the uploads/spreadsheet directory
        if os.path.exists("uploads/spreadsheet"):
            xlsx_files = [f for f in os.listdir("uploads/spreadsheet") if f.endswith((".xlsx", ".xls", ".csv"))]
            if xlsx_files:
                file_path = os.path.join("uploads/spreadsheet", xlsx_files[0])
                print(f"Found existing spreadsheet file: {file_path}")
    
    # Run test
    test_spreadsheet_conversion(file_path) 