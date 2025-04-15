import streamlit as st
import requests
import time
import json
from typing import List, Dict

# API endpoint (change to your backend server address)
API_URL = "http://localhost:8001"
def get_available_models() -> List[str]:
    """Fetch available models from the backend API"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        else:
            st.error(f"Error fetching models: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return []

def upload_document(file, selected_models):
    """Upload document to the backend API"""
    try:
        files = {"file": file}
        data = {"models": selected_models}
        response = requests.post(f"{API_URL}/upload", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading document: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

def get_session_status(session_id):
    """Get session status from the backend API"""
    try:
        response = requests.get(f"{API_URL}/session/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error checking session status: {response.text}")
            return {"status": "error"}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"status": "error"}

def query_document(session_id, question):
    """Query the document"""
    try:
        payload = {"session_id": session_id, "question": question}
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error querying document: {response.text}")
            return {"responses": {}}
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return {"responses": {}}

def main():
    st.set_page_config(
        page_title="Multi-Model RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Multi-Model Document Query System")
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "active_models" not in st.session_state:
        st.session_state.active_models = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
        
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Document Processing")
        
        if not st.session_state.document_processed:
            # Document upload
            uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], disabled=st.session_state.processing)
            
            if uploaded_file is not None:
                # Model selection
                available_models = get_available_models()
                
                if available_models:
                    selected_models = st.multiselect(
                        "Select models to use (you can select multiple)",
                        options=available_models,
                        default=[available_models[0]],
                        disabled=st.session_state.processing
                    )
                    
                    if st.button("Process Document", key="process_btn", disabled=not selected_models or st.session_state.processing):
                        if not selected_models:
                            st.error("Please select at least one model.")
                        else:
                            with st.spinner("Uploading and processing document... Please wait."):
                                result = upload_document(uploaded_file, selected_models)
                                
                                if result and "session_id" in result:
                                    st.session_state.session_id = result["session_id"]
                                    st.session_state.processing = True
                                    st.session_state.document_name = uploaded_file.name
                                    # Trigger rerun to update UI
                                    st.rerun()
                else:
                    st.error("Could not fetch available models from the backend.")
        else:
            # Display document info when processed
            st.success(f"Document processed: {st.session_state.document_name}")
            st.write(f"Active models: {', '.join(st.session_state.active_models)}")
            
            if st.button("Process new document", key="new_doc_btn"):
                # Reset session state
                st.session_state.session_id = None
                st.session_state.processing = False
                st.session_state.document_processed = False
                st.session_state.document_name = None
                st.session_state.active_models = []
                st.session_state.query_history = []
                st.rerun()
        
        # Display query history
        if st.session_state.query_history:
            st.header("Query History")
            for i, query in enumerate(st.session_state.query_history):
                if st.button(f"Q: {query[:30]}...", key=f"hist_{i}", disabled=st.session_state.processing):
                    # Fill the query input with the historical query
                    st.session_state.query_input = query
    
    # Main area for query interface
    if st.session_state.processing:
        # Check processing status
        with st.spinner("Processing document... This may take a moment."):
            status_info = get_session_status(st.session_state.session_id)
        
        if status_info["status"] == "processing":
            st.info("Document is being processed... Please wait.")
            
            # Auto-refresh
            time.sleep(2)
            st.rerun()
            
        elif status_info["status"] == "ready":
            if not st.session_state.document_processed:
                st.session_state.document_processed = True
                st.session_state.active_models = status_info["models"]
                # Show "Training Complete" popup effect
                st.success("🎉 Document processing complete! You can now ask questions.")
                st.balloons()  # Celebratory animation
            
            # Query interface
            st.header("Ask questions about your document")
            
            # Initialize query input in session state if not present
            if "query_input" not in st.session_state:
                st.session_state.query_input = ""
            
            # Create query form
            with st.form(key="query_form", clear_on_submit=True):
                query = st.text_input(
                    "Enter your question:",
                    value=st.session_state.query_input,
                    key="query_input",
                    disabled=not st.session_state.document_processed
                )
                submit_button = st.form_submit_button("Ask", disabled=not query)
                
                if submit_button and query:
                    with st.spinner("Searching document..."):
                        # Add query to history if it's new
                        if query not in st.session_state.query_history:
                            st.session_state.query_history.append(query)
                        
                        # Get responses
                        result = query_document(st.session_state.session_id, query)
                        
                        if result and "responses" in result:
                            # Display responses in tabs
                            tabs = st.tabs(st.session_state.active_models)
                            
                            for i, model in enumerate(st.session_state.active_models):
                                with tabs[i]:
                                    response = result["responses"].get(model, "No response received from this model.")
                                    st.markdown(response)
            
            # Display a prompt to ask questions if no query is entered yet
            if not query:
                st.info("Enter a question about the document to get started.")
                
        elif status_info["status"] == "error":
            st.error(f"Error processing document: {status_info.get('error', 'Unknown error')}")
            
            if st.button("Try again", key="retry_btn"):
                st.session_state.session_id = None
                st.session_state.processing = False
                st.rerun()
    else:
        # Initial page content when no document is being processed
        if not st.session_state.document_processed:
            st.info("👈 Upload a PDF document to begin.")
            
            # Example section
            with st.expander("How to use this application", expanded=True):
                st.markdown("""
                ### How to use this Document Query System:
                
                1. **Upload a PDF document** using the file uploader in the sidebar.
                2. **Select one or more models** to process and answer questions about your document.
                3. **Click "Process Document"** and wait for the processing to complete.
                4. **Ask questions** about your document in the query box.
                5. **Compare responses** from different models in the tabbed interface.
                
                This system uses Retrieval Augmented Generation (RAG) to provide accurate answers based on the content of your document.
                """)

if __name__ == "__main__":
    main()