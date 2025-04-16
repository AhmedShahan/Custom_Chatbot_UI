# RAG Chatbot with Multi-Document Support

This is a RAG (Retrieval Augmented Generation) chatbot that can process and answer questions about various document types.

## Supported Document Types

- PDF files (`.pdf`)
- PowerPoint presentations (`.ppt`, `.pptx`)
- Word documents (`.doc`, `.docx`)

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Start the backend server:

```
python main.py
```

This will start the FastAPI backend on port 8000.

### Start the Streamlit frontend:

```
streamlit run app.py
```

This will start the Streamlit frontend, typically on port 8501. Open your browser and navigate to the displayed URL.

## Using the Application

1. On the Training page, upload a document (PDF, PPT, PPTX, DOC, or DOCX).
2. Click "Process Document" to extract and index the content.
3. After processing, you'll be automatically redirected to the Playground page.
4. Use the chat interface to ask questions about the document.
5. You can switch between LLM and Non-LLM modes for different types of responses.

## Testing Different Document Types

You can use the included test script to verify functionality with different document types:

```
python test_document_types.py path/to/your/document.pdf
```

or test multiple documents at once:

```
python test_document_types.py doc1.pdf doc2.pptx doc3.docx
```

## Technical Details

The system uses:
- `unstructured` library for document parsing
- `faiss` for vector similarity search
- Various LLMs via Ollama for generating responses
- Different retrieval methods (rule-based, extractive, TF-IDF, etc.)

## Troubleshooting

- If you encounter issues with specific file formats, try converting them to a more recent version (e.g., .doc to .docx)
- For large documents, the processing might take some time
- Ensure you have sufficient memory for embedding and storing document contents 