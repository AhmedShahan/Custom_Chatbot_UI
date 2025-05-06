# IntelliDoc RAG Assistant

<div align="center">
  
<!-- ![IntelliDoc Logo](https://via.placeholder.com/200x100?text=IntelliDoc) -->

**Advanced Document Intelligence Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📚 Overview

IntelliDoc is an enterprise-grade Retrieval Augmented Generation (RAG) system designed for comprehensive document intelligence. It seamlessly processes multiple document formats, analyzes content with advanced algorithms, and delivers precise answers using state-of-the-art language models.

### Key Capabilities

- **Multi-document Processing**: Analyze dozens of documents simultaneously while maintaining cross-reference capabilities
- **Hybrid Retrieval Engine**: Combines vector similarity, keyword matching, and TF-IDF for superior accuracy
- **Adaptive Intelligence**: Choose between LLM-powered comprehensive answers or lightweight direct responses
- **Interactive UI**: Intuitive interface for document upload, question-answering, and result visualization
- **Browser Integration**: Chrome extension for accessing your document knowledge base from anywhere

---

## 🏗️ Architecture
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
IntelliDoc employs a modern, modular architecture designed for scalability and performance.

### Document Processing Pipeline

<details>
<summary><b>Click to expand detailed workflow</b></summary>

1. **Document Ingestion**
   - Secure document upload through FastAPI endpoint
   - Initial validation and format detection
   - Document queuing for parallel processing

2. **Text Extraction**
   - Format-specific extraction using the unstructured library
   - Preservation of document structure and relationships
   - Special handling for tables, images, and complex layouts

3. **Advanced Processing**
   - Intelligent chunking with overlap for context preservation
   - Metadata extraction and structural mapping
   - Entity recognition and relationship identification

4. **Knowledge Engineering**
   - Embedding generation using transformer models
   - Multi-dimensional indexing for rapid retrieval
   - Hierarchical representation of document knowledge

</details>

### Query Processing Pipeline

<details>
<summary><b>Click to expand detailed workflow</b></summary>

1. **Query Analysis**
   - Intent recognition and classification
   - Key concept extraction and expansion
   - Query reformulation for optimal retrieval

2. **Multi-strategy Retrieval**
   - **Vector Search**: Semantic matching using dense embeddings
   - **Keyword Matching**: Exact and fuzzy matching for precision
   - **TF-IDF Analysis**: Statistical relevance scoring
   - **Hybrid Approach**: Weighted combination of all methods

3. **Response Generation**
   - Context assembly and relevance ranking
   - Answer synthesis with citation tracking
   - Response formatting and delivery

</details>

---

## 🔧 Technology Stack

IntelliDoc leverages cutting-edge technologies across its stack:

### Core Technologies

| Category                        | Technologies       |
|---------------------------------|--------------------|
| <th colspan="2">**Cloud Infrastructure & DevOps**</th> |
|**Version Control System**|Git|
|**Version Control Host**|GitHub|
| <th colspan="2">**Backend Technology**</th> |
|**Programming Language**|Python 3.10.12|
|**Backend Framework**|FastAPI|
| <th colspan="2">**AI COMPONENTS**</th> |
|**Eco System**|LangChain|
|**Deep Learning framework**|Pytorch|
|**Document management & Serialization**|UUID & JSON|
|**Document split**|unstructured.io (https://unstructured.io/)|
|**Document Conversion**|python-docx, python-pptx, pandas, BeautifulSoup4|
|**Text Embedding**|RoBERTa, OpenAI's text-embedding-ada-002 |
|**Sentence Transformer**|paraphrase-T5-large, paraphrase-roberta-large-v1|
|**Vector Store:**|Facebook AI Similarity Search (FAISS)|
|**Generation Context**|- Non LLM Based: Knowledge Graph-Based Responses, Extractive Summarization, rule|
|**LLM Based**|DeepSeek-r1|
|**Web Scraping**| Crawl4ai, BeautifulSoup, requests, Selenium|
|**Web Content loader**|WebBaseLoader, Selenium|


### Supported Document Types

<details>
<summary><b>Click to view all supported formats</b></summary>

- **PDF Files** (`.pdf`)
  - Text extraction with layout preservation
  - Table detection and structured extraction
  - Image identification and processing
  - Form field recognition

- **Office Documents**
  - **Word** (`.doc`, `.docx`): Full text and structure processing
  - **PowerPoint** (`.ppt`, `.pptx`): Slide content with presenter notes
  - **Excel/CSV** (`.csv`, `.xls`, `.xlsx`): Tabular data with context

- **Web Content**
  - Direct URL processing
  - HTML structure preservation
  - Dynamic content handling capabilities

- **Plain Text** (`.txt`)
  - Basic processing with paragraph structure
  - Support for markdown formatting

</details>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended for large documents)
- 10GB free disk space
- Internet connection for initial setup

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/intellidoc-rag-assistant.git
cd intellidoc-rag-assistant
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install system dependencies**

<details>
<summary>Linux</summary>

```bash
# PDF processing tools
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr

# For spaCy language model
python -m spacy download en_core_web_sm

# For Ollama
curl -fsSL https://ollama.com/install.sh | sh
```
</details>

<details>
<summary>macOS</summary>

```bash
# Using Homebrew
brew install poppler tesseract

# For spaCy language model
python -m spacy download en_core_web_sm

# For Ollama
curl -fsSL https://ollama.com/install.sh | sh
```
</details>

<details>
<summary>Windows</summary>

```powershell
# For spaCy language model
python -m spacy download en_core_web_sm

# For Ollama - Download from https://ollama.com/download
```

Install Poppler and Tesseract manually:
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
</details>

5. **Install required Ollama models**

```bash
ollama pull deepseek-r1:14b
```

### Running IntelliDoc

1. **Start the backend server**

```bash
python main.py
```

2. **In a new terminal, start the frontend**

```bash
streamlit run app.py
```

3. **Access the application**

Open your browser and navigate to: `http://localhost:8501`

---

## 💡 User Guide

### Document Processing

<details>
<summary><b>Step-by-step instructions</b></summary>

1. Navigate to the **Training** page in the sidebar
2. Upload your documents using the file uploader
   - Drag and drop multiple files
   - Or click to select files from your computer
3. Click **Process Documents** to begin extraction
4. View the processing status in real-time
5. Once complete, you'll be automatically redirected to the Playground

**Pro Tip**: For best results with scanned PDFs, enable OCR processing in the advanced settings.
</details>

### Asking Questions

<details>
<summary><b>Getting the best answers</b></summary>

1. Navigate to the **Playground** page
2. Type your question in the chat interface
3. Select your preferred answer mode:
   - **LLM Mode**: Comprehensive, natural language answers
   - **Non-LLM Mode**: Direct, extractive responses
4. Click **Ask** or press Enter
5. Review the answer, including source citations
6. Follow up with related questions to explore further

**Query Tips**:
- Be specific in your questions
- Include key terms from the documents
- For complex topics, break down into multiple questions
- Use the "explain" command for detailed breakdowns of concepts
</details>

### Advanced Features

<details>
<summary><b>Power user capabilities</b></summary>

- **Document Explorer**: Browse document structure and content
- **Table Extraction**: View and export tables from documents
- **Web Content Processing**: Analyze content directly from URLs
- **Visualization Tools**: See document relationships and key concepts
- **Custom Retrieval**: Fine-tune the retrieval parameters
- **Export Functionality**: Save conversations and findings

**Command Syntax**:
- `/help` - Show available commands
- `/summary [document_name]` - Generate document summary
- `/extract [table_number]` - Extract specific table
- `/compare [doc1] [doc2]` - Compare two documents
- `/search [keyword]` - Search across all documents
</details>

---

## 🔌 Chrome Extension

Integrate IntelliDoc directly into your browsing experience with our Chrome extension.

### Installation

<details>
<summary><b>Step-by-step installation guide</b></summary>

1. Locate the `chrome-extension` directory in the project
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top-right)
4. Click "Load unpacked" and select the `chrome-extension` folder
5. Pin the extension to your toolbar for easy access

**Configuration**:
1. Click the IntelliDoc icon in your toolbar
2. Enter your IntelliDoc server URL in the settings
3. Customize appearance and behavior preferences
4. Save settings and refresh your browser
</details>

### Features

- **Universal Access**: Chat with your documents from any webpage
- **Context Awareness**: Automatically suggests relevant document content based on current webpage
- **Knowledge Integration**: Combine web browsing with your document knowledge base
- **Seamless Experience**: Responsive design that adapts to your viewport

---

## ⚙️ Configuration Options

IntelliDoc can be customized through environment variables or a configuration file:

### Core Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `MODEL_NAME` | Default LLM model | `"deepseek-r1:14b"` |
| `EMBEDDING_MODEL` | Model for embeddings | `"roberta-base"` |
| `TOP_K_RESULTS` | Number of passages to retrieve | `5` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `HYBRID_ALPHA` | Vector/keyword search weight | `0.7` |

### Advanced Configuration

Create a `.env` file in the project root or set environment variables directly:

```bash
# Server configuration
PORT=8000
WORKERS=4
LOG_LEVEL=INFO

# Processing settings
ENABLE_OCR=true
MAX_DOCUMENT_SIZE=25
DOCUMENT_CACHE_TTL=3600

# LLM settings
MODEL_NAME=deepseek-r1:14b
TEMPERATURE=0.7
MAX_TOKENS=2048

# Security settings
API_KEY_REQUIRED=false
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

<details>
<summary><b>Document Processing Issues</b></summary>

**Problem**: PDF text extraction fails
- **Solution**: Ensure the PDF contains extractable text (not scanned)
- **Alternative**: Enable OCR processing in advanced settings
- **Verification**: Check if you can select text in your PDF viewer

**Problem**: Large documents cause memory errors
- **Solution**: Increase chunk size in advanced settings
- **Alternative**: Split document into smaller files
- **System Solution**: Increase system RAM or use swap space
</details>

<details>
<summary><b>Model Loading Issues</b></summary>

**Problem**: Ollama model fails to load
- **Solution**: Ensure Ollama is running (`ollama serve`)
- **Verification**: Check available models (`ollama list`)
- **Alternative**: Try a smaller model if memory is limited

**Problem**: Slow initial loading
- **Note**: First-time model loading caches files to disk
- **Solution**: Be patient during first load, subsequent loads are faster
- **Optimization**: Use SSD storage for faster loading
</details>

<details>
<summary><b>Answer Quality Issues</b></summary>

**Problem**: Irrelevant or incorrect answers
- **Solution**: Try reformulating your question
- **Alternative**: Adjust retrieval settings (increase `TOP_K_RESULTS`)
- **Advanced**: Try different retrieval methods (hybrid recommended)

**Problem**: Missing information in answers
- **Solution**: Check document processing status
- **Verification**: Browse document in Document Explorer
- **Alternative**: Split complex questions into simpler ones
</details>

### Detailed Logs

For deeper troubleshooting, enable detailed logging:

```bash
# Run with debug logging
python main.py --log-level debug

# Save logs to file
python main.py --log-file intellidoc.log
```

---

## 🧪 Testing

Validate IntelliDoc's functionality with the included test suite:

```bash
# Test all components
pytest

# Test specific document types
python test_document_types.py path/to/document.pdf

# Performance testing with large documents
python benchmark.py --file large_document.pdf
```

### Document Type Compatibility Test

Test multiple documents simultaneously:

```bash
python test_document_types.py doc1.pdf doc2.pptx doc3.docx
```

---

## 🤝 Contributing

We welcome contributions to IntelliDoc! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request with detailed description

---


## 📊 System Architecture Deep Dive

### Vectorization Strategies

IntelliDoc employs sophisticated vectorization techniques:

<details>
<summary><b>Dual Encoding Architecture</b></summary>

Our system uses separate vector spaces for different content aspects:

1. **Semantic Vectors**
   - RoBERTa-powered dense embeddings
   - Captures meaning and context
   - Optimized for conceptual similarity

2. **Lexical Vectors**
   - TF-IDF representations
   - Captures keyword specificity
   - Handles technical terminology precisely

3. **Structural Vectors**
   - Document structure representations
   - Maintains hierarchical relationships
   - Preserves section and subsection context

The system dynamically weights these vector spaces based on query characteristics.
</details>

### Retrieval Optimization

<details>
<summary><b>Multi-stage Retrieval Pipeline</b></summary>

IntelliDoc uses a cascading retrieval approach:

1. **Initial Candidate Selection**
   - Fast FAISS index query to identify potential matches
   - Document-level filtering based on metadata

2. **Fine-grained Passage Retrieval**
   - Context-aware chunk selection
   - Re-ranking based on query-passage relevance
   - Diversity sampling to avoid redundancy

3. **Context Assembly**
   - Dynamic window expansion around key passages
   - Cross-reference resolution between documents
   - Sequential context preservation for narrative content

This multi-stage approach balances speed and accuracy.
</details>

---

<div align="center">

**[Documentation](https://github.com/yourusername/intellidoc-rag-assistant/wiki)** | 
**[Issues](https://github.com/yourusername/intellidoc-rag-assistant/issues)** | 
**[Roadmap](https://github.com/yourusername/intellidoc-rag-assistant/projects)**

</div>
