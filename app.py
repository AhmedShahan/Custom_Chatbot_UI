import streamlit as st
import requests
import time
import validators  # Make sure to install this package
import base64
from datetime import datetime
from fpdf import FPDF
import socket
import zipfile
import os
import io

# Configure page
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Welcome"
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "extracted_tables" not in st.session_state:
    st.session_state.extracted_tables = []

# Helper function to get the current server URL
def get_server_url():
    # Return fixed URL for simplicity and consistency
    return "http://0.0.0.1:8501/"

# Helper function to create a zip file of the Chrome extension
def create_extension_zip():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Get all files in the chrome-extension directory
        for root, dirs, files in os.walk("chrome-extension"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "chrome-extension")
                zipf.write(file_path, arcname=arcname)
    
    memory_file.seek(0)
    return memory_file.getvalue()

def upload_document(file):
    """Upload document to backend"""
    try:
        files = {"file": file}
        response = requests.post("http://localhost:8000/upload-document", files=files)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        try:
            result = response.json()
            if "error" in result:
                return {"error": result["error"]}
            else:
                return {
                    "message": result["message"],
                    "doc_count": result.get("doc_count", 0)
                }
        except ValueError as e:
            # Handle case where response is not valid JSON
            return {"error": f"Invalid server response: {response.text[:100]}..."}
    except requests.RequestException as e:
        # Handle network-related errors
        return {"error": f"Request failed: {str(e)}"}

def process_website(url):
    """Process website URL"""
    try:
        response = requests.post(
            "http://localhost:8000/process-website",
            params={"url": url}
        )
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        try:
            result = response.json()
            if "error" in result:
                return {"error": result["error"]}
            else:
                return {
                    "message": result["message"],
                    "doc_count": result.get("doc_count", 0)
                }
        except ValueError as e:
            # Handle case where response is not valid JSON
            return {"error": f"Invalid server response: {response.text[:100]}..."}
    except requests.RequestException as e:
        # Handle network-related errors
        return {"error": f"Request failed: {str(e)}"}

def get_document_status():
    """Get document status from backend"""
    try:
        response = requests.get("http://localhost:8000/document-status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to get document status: {str(e)}"}

def get_document_content():
    """Get full document content from backend"""
    try:
        response = requests.get("http://localhost:8000/document-content")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to get document content: {str(e)}"}

def get_extracted_tables():
    """Get extracted tables from backend"""
    try:
        response = requests.get("http://localhost:8000/extracted-tables")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to get extracted tables: {str(e)}"}

def get_answer(question, model_type, model_name):
    """Get answer from backend"""
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            params={
                "question": question,
                "model_type": model_type,
                "model_name": model_name
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def create_chat_pdf(messages):
    """Create a PDF from chat history"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Chat History", ln=True, align="C")
    pdf.ln(10)
    
    # Add timestamp
    pdf.set_font("Arial", "I", 10)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(190, 5, f"Generated on: {current_time}", ln=True)
    pdf.ln(5)
    
    # Add chat messages
    pdf.set_font("Arial", "", 12)
    for i, message in enumerate(messages):
        role = message["role"].upper()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 10, f"{role}:", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Handle multi-line content and encode properly to avoid Unicode errors
        content = message["content"]
        
        # Replace problematic Unicode characters with ASCII equivalents
        content = content.replace(''', "'").replace(''', "'")
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace('—', '-').replace('–', '-')
        content = content.replace('…', '...')
        
        # Remove any remaining non-Latin1 characters
        content = ''.join(c if ord(c) < 256 else '_' for c in content)
        
        pdf.multi_cell(190, 8, content)
        pdf.ln(5)
    
    # Return PDF as base64 string using a BytesIO buffer to avoid encoding issues
    try:
        pdf_output = pdf.output(dest="S").encode("latin1")
        return base64.b64encode(pdf_output).decode("utf-8")
    except UnicodeEncodeError:
        # Fallback if encoding still fails
        pdf.set_font("Arial", "B", 12)
        pdf.add_page()
        pdf.cell(190, 10, "Error: Could not encode all characters in the chat.", ln=True)
        pdf_output = pdf.output(dest="S").encode("latin1")
        return base64.b64encode(pdf_output).decode("utf-8")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Welcome Page"):
        st.session_state.current_page = "Welcome"
    if st.button("Training Page"):
        st.session_state.current_page = "Training"
    if st.button("Playground"):
        if not st.session_state.document_processed:
            st.error("Please process a document first!")
        else:
            st.session_state.current_page = "Playground"

# Welcome Page
if st.session_state.current_page == "Welcome":
    st.title("Welcome to RAG Chatbot")
    
    # Description
    st.markdown("""
    This application provides a powerful RAG (Retrieval-Augmented Generation) chatbot that can answer questions based on your documents.
    You can upload documents or provide website URLs and then interact with your data through a smart conversational interface.
    """)
    
    # Get the current Streamlit URL
    iframe_url = get_server_url()
    
    # Copy to clipboard functionality using HTML and JavaScript
    st.subheader("Embed the Chatbot in Your Website")
    st.markdown(f"**Iframe URL:** `{iframe_url}`")
    
    copy_button_html = f"""
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <input type="text" value="{iframe_url}" 
               id="iframe-url" 
               style="flex-grow: 1; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;" 
               readonly>
        <button onclick="copyUrl()" 
                style="margin-left: 0.5rem; background-color: #4CAF50; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;">
            Copy URL
        </button>
    </div>
    
    <script>
    function copyUrl() {{
        var copyText = document.getElementById("iframe-url");
        copyText.select();
        copyText.setSelectionRange(0, 99999);
        navigator.clipboard.writeText(copyText.value);
        
        var button = document.querySelector("button");
        var originalText = button.innerHTML;
        button.innerHTML = "Copied!";
        setTimeout(function() {{
            button.innerHTML = originalText;
        }}, 2000);
    }}
    </script>
    """
    
    st.markdown(copy_button_html, unsafe_allow_html=True)
    
    # Chrome Extension Download Button
    st.subheader("Use Our Chrome Extension")
    st.markdown("""
    Install our Chrome Extension to access the chatbot from any webpage. 
    Just click the button below to download the extension, then:
    
    1. Unzip the downloaded file
    2. Open Chrome and go to `chrome://extensions/`
    3. Enable "Developer mode" (toggle in the top-right)
    4. Click "Load unpacked" and select the unzipped folder
    5. The extension icon will appear in your toolbar
    """)
    
    # Generate zip file for download
    extension_zip = create_extension_zip()
    st.download_button(
        label="Download Chrome Extension",
        data=extension_zip,
        file_name="rag-chatbot-extension.zip",
        mime="application/zip",
        help="Download the Chrome extension package as a zip file"
    )
    
    # Get Started Button
    st.markdown("---")
    if st.button("Get Started →", type="primary"):
        st.session_state.current_page = "Training"
        st.rerun()

# Training Page
elif st.session_state.current_page == "Training":
    st.title("Document Training")
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Document", "Website URL"])
    
    with tab1:
        # Document upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a document", type=["pdf", "ppt", "pptx", "doc", "docx", "csv", "xls", "xlsx"])
        
        if uploaded_file:
            if st.button("Process Document", key="process_doc"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = upload_document(uploaded_file)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"{result['message']} Found {result.get('doc_count', 0)} sections in the document.")
                        st.session_state.document_processed = True
                        st.session_state.current_page = "Playground"
    
    with tab2:
        # Website URL input section
        st.subheader("Process Website Content")
        website_url = st.text_input("Enter website URL", placeholder="https://example.com")
        
        # Validate URL when input changes
        if website_url:
            if not website_url.startswith(('http://', 'https://')):
                website_url = 'https://' + website_url
                st.info(f"Added https:// prefix: {website_url}")
            
            if not validators.url(website_url):
                st.error("Please enter a valid URL")
            else:
                st.info("Valid URL format. Click 'Process Website' to continue.")
        
        if website_url and validators.url(website_url):
            if st.button("Process Website", key="process_website"):
                with st.spinner(f"Processing website: {website_url}... This may take a few minutes depending on the size of the website."):
                    result = process_website(website_url)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"{result['message']} Found {result.get('doc_count', 0)} sections from the website.")
                        st.session_state.document_processed = True
                        st.session_state.current_page = "Playground"
    
    # Document status section
    st.subheader("Current Document Status")
    if st.button("Check Document Status"):
        with st.spinner("Checking document status..."):
            status = get_document_status()
            if "error" in status:
                st.error(status["error"])
            elif status["status"] == "No document loaded":
                st.info("No document is currently loaded.")
            else:
                st.info(f"Document loaded with {status['document_count']} sections.")
                if "documents" in status and status["documents"]:
                    st.write("Document sections:")
                    for doc in status["documents"]:
                        st.write(f"- {doc['title']} ({doc['text_length']} chars)")

# Playground Page
elif st.session_state.current_page == "Playground":
    st.title("RAG Chatbot Playground")

    # Create tabs for different views - removing content_tab and tables_tab
    playground_tab = st.container()
    
    with playground_tab:
        # Model Selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.radio("Select Model Type", ["LLM", "Non-LLM"])
        
        with col2:
            if model_type == "LLM":
                model_name = "deepseek-r1:latest"
                st.info("Using deepseek-r1:latest model")
            else:
                model_name = st.selectbox(
                    "Select Method",
                    ["rule", "extractive", "extraction", "tfidf"]
                )

        # Display current configuration
        st.info(f"Current Configuration: {model_type} - {model_name}")

        # Chat Interface
        st.subheader("Chat Interface")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the document"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_answer(prompt, model_type, model_name)
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        # Create a placeholder for streaming response
                        message_placeholder = st.empty()
                        full_response = response["answer"]
                        
                        # Stream the response character by character
                        displayed_response = ""
                        for char in full_response:
                            displayed_response += char
                            message_placeholder.markdown(displayed_response + "▌")
                            time.sleep(0.005)  # Small delay for streaming effect
                            
                        # Final display without cursor
                        message_placeholder.markdown(displayed_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Chat export options
        if st.session_state.messages:
            if st.button("Export Chat to PDF"):
                pdf_base64 = create_chat_pdf(st.session_state.messages)
                
                # Create download link
                href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="chat_history.pdf">Download Chat History PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun() 