# IntelliDoc RAG Assistant

IntelliDoc is an advanced Retrieval Augmented Generation (RAG) chatbot system designed to process, analyze, and answer questions about multiple document types. It combines state-of-the-art document processing with advanced language models to provide accurate and contextually relevant answers.

## Overview

IntelliDoc uses a hybrid retrieval system that combines vector similarity search, keyword matching, and TF-IDF to find the most relevant content from your documents. It then uses a large language model to generate natural, comprehensive answers based on the retrieved content. The system can work with multiple documents simultaneously, maintaining context across different sources.

## Architecture and Workflow

### System Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Document Input │────▶│ Document Processor │────▶│ Text Extraction │
└─────────────────┘     └───────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│    Response     │◀────│  LLM Generation   │◀────│   Vectorization │
│   Generation    │     │  or Rule-based    │     │    & Indexing   │
└─────────────────┘     └───────────────────┘     └─────────────────┘
        │                                                  ▲
        │                                                  │
        ▼                                                  │
┌─────────────────┐     ┌───────────────────┐             │
│  User Interface │────▶│   Query Analysis  │─────────────┘
└─────────────────┘     └───────────────────┘
```

### Document Processing Workflow

1. **Document Ingestion**: Upload documents through FastAPI endpoint
2. **Text Extraction**: Use unstructured library to extract text, tables, and structure
3. **Chunking & Processing**: Documents are split into manageable chunks
4. **Feature Extraction**: Extract keywords, entities, and metadata
5. **Vectorization**: Generate embeddings using transformer models
6. **Indexing**: Store embeddings in FAISS vector indexes and build TF-IDF representations

### Query Processing Workflow

1. **Query Analysis**: Extract keywords and intent from user query
2. **Multi-strategy Retrieval**:
   - Vector similarity search (semantic matching)
   - Keyword matching
   - TF-IDF similarity
   - Hybrid combinations of the above
3. **Response Generation**:
   - LLM Mode: Use retrieved content to generate comprehensive answers
   - Non-LLM Mode: Use rule-based or extractive methods for direct answers

### Vectorization Strategies

IntelliDoc employs multiple vectorization approaches:

1. **Transformer Embeddings**: Uses RoBERTa model to create dense vector representations of text
2. **Dual Encoding**: Separate encodings for titles and content for more precise matching
3. **TF-IDF Vectorization**: Term frequency-inverse document frequency for keyword relevance
4. **FAISS Indexing**: Fast similarity search using FAISS for efficient vector retrieval
5. **Keyword Extraction**: Supplementary representation for exact matching scenarios

## Supported Document Types

IntelliDoc supports the following document formats:

- **PDF Files** (`.pdf`): Full support with text, table, and image extraction
- **PowerPoint Presentations** (`.ppt`, `.pptx`): Extracts slides, notes, and structure
- **Word Documents** (`.doc`, `.docx`): Processes text, tables, and document structure
- **Spreadsheets** (`.csv`, `.xls`, `.xlsx`): Processes tabular data with row/column context
- **Web Content** (via URL): Extracts content from websites with structure preservation
- **Plain Text** (`.txt`): Basic text processing

## Technology Stack

### Core Libraries and Frameworks
- **Python 3.10+**: Base programming language
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend user interface
- **Langchain**: Framework for LLM application building
- **Ollama**: Local LLM integration

### Document Processing
- **Unstructured**: Document parsing and extraction library
- **PyPDF2/pikepdf**: PDF processing
- **python-docx**: Word document processing
- **python-pptx**: PowerPoint processing
- **pandas**: Spreadsheet and tabular data handling
- **BeautifulSoup4**: Web scraping and HTML processing

### Vector Search and Embeddings
- **FAISS**: Vector similarity search
- **Transformers**: Hugging Face transformer models for embeddings
- **RoBERTa**: Pre-trained model for text embeddings
- **scikit-learn**: For TF-IDF vectorization
- **numpy**: Numerical operations for vector manipulation

### Natural Language Processing
- **spaCy**: NLP toolkit for entity recognition and text processing
- **NLTK**: Natural Language Toolkit for text tokenization and analysis

### LLM Integration
- **deepseek-r1**: Default LLM model through Ollama
- **Optional support for other Ollama models**

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/intellidoc-rag-assistant.git
   cd intellidoc-rag-assistant
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install additional system dependencies (if needed):
   ```
   # For PDF processing
   apt-get install -y poppler-utils

   # For spaCy language model
   python -m spacy download en_core_web_sm

   # For Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh
   ```

4. Install required Ollama models:
   ```
   ollama pull deepseek-r1:14b
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

1. **Document Training**:
   - Navigate to the Training page
   - Upload one or more documents (PDF, PPT, DOCX, etc.)
   - Click "Process Document" to extract and index the content
   - Wait for the processing to complete (time depends on document size)

2. **Chatting with Documents**:
   - After processing, you'll be redirected to the Playground page
   - Type your question in the chat interface
   - Select LLM mode for comprehensive answers or Non-LLM mode for direct responses
   - Choose your preferred model or response method
   - Click "Ask" or press Enter to submit your question

3. **Model and Method Selection**:
   - **LLM Mode**: Uses chosen language model to generate comprehensive answers
   - **Non-LLM Mode**: Choose from:
     - Rule-based: Pattern matching for direct answers
     - Extractive: Returns relevant document snippets
     - TF-IDF: Uses term frequency for relevant passage retrieval
     - Hybrid: Combines multiple methods for optimal retrieval

4. **Additional Features**:
   - View document content and structure
   - Extract tables from documents
   - Process web content from URLs
   - Visualize document sections and metadata

## Troubleshooting

- **PDF Processing Issues**:
  - Ensure PDF is not scanned or contains extractable text
  - Try converting to a newer PDF version if having issues
  - For scanned PDFs, OCR processing may be required

- **Memory Issues with Large Documents**:
  - Try processing documents in smaller chunks
  - Increase system memory if processing very large documents
  - Use smaller embedding models if experiencing out-of-memory errors

- **Model Loading Errors**:
  - Ensure Ollama is running (`ollama serve`)
  - Check that required models are installed (`ollama list`)
  - Try using a smaller model if facing resource constraints

- **Slow Processing**:
  - Document size affects processing time, be patient with large files
  - First-time model loading may take longer
  - Consider using CPU-only mode if GPU memory is limited

- **Incorrect or Unrelated Answers**:
  - Try reformulating your question
  - Check that the document was processed properly
  - Try different retrieval methods (hybrid often works best)
  - Adjust the number of retrieved passages in advanced settings

## Advanced Configuration

The system can be customized through environment variables or configuration files:

- `MODEL_NAME`: Default LLM model (e.g., "deepseek-r1:14b")
- `EMBEDDING_MODEL`: Model for generating embeddings
- `TOP_K_RESULTS`: Number of passages to retrieve (default: 5)
- `CHUNK_SIZE`: Document chunk size for processing
- `HYBRID_ALPHA`: Weight between vector and keyword search in hybrid mode

## Chrome Extension Integration

IntelliDoc includes a Chrome extension that allows you to access the chatbot directly from any web page.

### Extension Installation

1. Navigate to the `chrome-extension` directory in the project
2. Open Chrome and go to `chrome://extensions/`
3. Enable "Developer mode" by toggling the switch in the top-right corner
4. Click "Load unpacked" and select the `chrome-extension` folder from the project
5. The extension should now appear in your Chrome toolbar

### Using the Extension

1. Click the IntelliDoc icon in your Chrome toolbar to toggle the chat bubble on the current page
2. A blue chat bubble will appear in the bottom-right corner of the page
3. Click the bubble to open the chatbot interface
4. On first use, enter the URL where your IntelliDoc instance is running (e.g., `http://localhost:8501`)
5. The chatbot will open in an iframe on the current page
6. You can now interact with your documents without leaving the current website

### Extension Features

- **Web Page Integration**: Access your documents and ask questions while browsing
- **Persistent Settings**: The extension remembers your chatbot URL
- **Cross-Origin Support**: Works with most websites (see troubleshooting for HTTPS limitations)
- **Responsive Design**: Adapts to different screen sizes

### Extension Troubleshooting

- **HTTPS Security Issues**: If using the extension on HTTPS sites, your IntelliDoc server must also use HTTPS, or you'll need to use the "Open in New Tab" option
- **CORS Errors**: If you see loading errors, ensure your Streamlit server has CORS disabled:
  ```
  streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false
  ```
- **Extension Not Working**: Ensure the content script is enabled and try refreshing the page
- **For detailed troubleshooting**: See `chrome-extension/TROUBLESHOOTING.md`

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
