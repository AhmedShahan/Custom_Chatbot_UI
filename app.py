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
    st.session_state.pdf_processed = False

def upload_pdf(file):
    """Upload PDF to backend"""
    files = {"file": file}
    response = requests.post("http://localhost:8000/upload-pdf", files=files)
    return response.json()

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
        if not st.session_state.pdf_processed:
            st.error("Please process a PDF first!")
        else:
            st.session_state.current_page = "Playground"

# Training Page
if st.session_state.current_page == "Training":
    st.title("PDF Training")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                result = upload_pdf(uploaded_file)
                st.success(result["message"])
                st.session_state.pdf_processed = True
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
    if prompt := st.chat_input("Ask a question about the PDF"):
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