import streamlit as st
import requests
import time

# Configure page
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Training"
if "pdf_processed" not in st.session_state:
    st.session_state.document_processed = False

def upload_document(file):
    """Upload document to backend"""
    try:
        files = {"file": file}
        response = requests.post("http://localhost:8000/upload-document", files=files)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        try:
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            return {"error": f"Invalid server response: {response.text[:100]}..."}
    except requests.RequestException as e:
        # Handle network-related errors
        return {"error": f"Request failed: {str(e)}"}

def get_answer(question, model_type, model_name):
    """Get answer from backend"""
    response = requests.post(
        "http://localhost:8000/ask",
        params={
            "question": question,
            "model_type": model_type,
            "model_name": model_name
        }
    )
    return response.json()

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Training Page"):
        st.session_state.current_page = "Training"
    if st.button("Playground"):
        if not st.session_state.document_processed:
            st.error("Please process a document first!")
        else:
            st.session_state.current_page = "Playground"

# Training Page
if st.session_state.current_page == "Training":
    st.title("Document Training")
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "ppt", "pptx", "doc", "docx"])
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                result = upload_document(uploaded_file)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(result["message"])
                    st.session_state.document_processed = True
                    st.session_state.current_page = "Playground"

# Playground Page
elif st.session_state.current_page == "Playground":
    st.title("RAG Chatbot Playground")

    # Model Selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio("Select Model Type", ["LLM", "Non-LLM"])
    
    with col2:
        if model_type == "LLM":
            model_name = st.selectbox(
                "Select LLM Model",
                ["llama3.2", "gamma", "opencoder:1.5b","chatglm-6b-v2"]
            )
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
                    st.write(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun() 