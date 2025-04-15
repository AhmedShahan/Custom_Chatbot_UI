import streamlit as st
import pandas as pd
from datetime import datetime
import os
import shutil
import tempfile

# আপনার RAG সিস্টেম সরাসরি ইম্পোর্ট করুন
from rag_system import RAGSystem

st.set_page_config(page_title="RAG চ্যাটবট", page_icon="📚", layout="wide")

# ফোল্ডারগুলি তৈরি করুন
UPLOAD_DIR = "uploaded_pdfs"
MODEL_DIR = "saved_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# অ্যাপে স্টেট ম্যানেজমেন্ট
if "active_rag" not in st.session_state:
    st.session_state.active_rag = None

# হেডার এবং শিরোনাম
st.title("PDF RAG চ্যাটবট")
st.subheader("আপনার PDF থেকে প্রশ্ন জিজ্ঞাসা করুন")

# সাইডবার সেটআপ
with st.sidebar:
    st.header("সেটিংস")
    
    # PDF আপলোড বিভাগ
    st.subheader("নতুন PDF আপলোড করুন")
    uploaded_file = st.file_uploader("PDF নির্বাচন করুন", type="pdf")
    
    model_name = st.selectbox(
        "LLM মডেল নির্বাচন করুন",
        ["gemma3:latest", "deepseek-r1", "llama3:latest", "llama3:70b"],
        index=0
    )
    
    use_llm = st.checkbox("LLM ব্যবহার করুন", value=True)
    
    method = st.selectbox(
        "প্রশ্ন উত্তর পদ্ধতি",
        ["hybrid", "rule", "extractive", "extraction", "tfidf"],
        index=0
    )
    
    if uploaded_file and st.button("আপলোড এবং প্রশিক্ষণ শুরু করুন"):
        with st.spinner("PDF আপলোড এবং প্রশিক্ষণ চলছে..."):
            # সাময়িক ফাইল সংরক্ষণ
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file, buffer)
            
            try:
                # RAG সিস্টেম তৈরি ও প্রশিক্ষণ
                st.session_state.active_rag = RAGSystem(model_name=model_name, use_llm=use_llm, method=method)
                st.session_state.active_rag.ingest_pdf(file_path)
                
                # মডেল সংরক্ষণ
                model_path_prefix = os.path.join(MODEL_DIR, f"{os.path.splitext(uploaded_file.name)[0]}")
                st.session_state.active_rag.save(model_path_prefix)
                
                st.success("PDF আপলোড এবং প্রশিক্ষণ সম্পন্ন!")
            except Exception as e:
                st.error(f"ত্রুটি: {str(e)}")
    
    # সংরক্ষিত মডেল লোড
    st.subheader("সংরক্ষিত মডেল লোড করুন")
    
    if st.button("সংরক্ষিত মডেল তালিকা"):
        # ডিরেক্টরিতে উপস্থিত ফাইলগুলি খুঁজে বের করা
        model_files = {}
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith("_documents.json"):
                model_name = filename.replace("_documents.json", "")
                model_files[model_name] = os.path.getmtime(os.path.join(MODEL_DIR, filename))
        
        # তালিকা সৃষ্টি
        models = [
            {"name": name, "created": timestamp}
            for name, timestamp in model_files.items()
        ]
        
        # সময় অনুযায়ী সাজানো
        models.sort(key=lambda x: x["created"], reverse=True)
        
        if models:
            model_df = pd.DataFrame([
                {
                    "নাম": m["name"],
                    "তৈরির সময়": datetime.fromtimestamp(m["created"]).strftime("%Y-%m-%d %H:%M")
                }
                for m in models
            ])
            st.dataframe(model_df)
            
            selected_model = st.selectbox("মডেল নির্বাচন করুন", [m["name"] for m in models])
            
            if st.button("মডেল লোড করুন"):
                with st.spinner("মডেল লোড হচ্ছে..."):
                    try:
                        model_path_prefix = os.path.join(MODEL_DIR, selected_model)
                        st.session_state.active_rag = RAGSystem.load(
                            model_path_prefix, 
                            model_name=model_name,
                            use_llm=use_llm,
                            method=method
                        )
                        st.success(f"মডেল '{selected_model}' সফলভাবে লোড হয়েছে!")
                    except Exception as e:
                        st.error(f"ত্রুটি: {str(e)}")
        else:
            st.info("কোন সংরক্ষিত মডেল পাওয়া যায়নি")

# মূল অংশে চ্যাট ইন্টারফেস
st.header("PDF চ্যাটবট")

# চ্যাট ইতিহাস সংরক্ষণ
if "messages" not in st.session_state:
    st.session_state.messages = []

# আগের বার্তাগুলি প্রদর্শন
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ব্যবহারকারীর ইনপুট প্রক্রিয়া
prompt = st.chat_input("আপনার প্রশ্ন জিজ্ঞাসা করুন...")
if prompt:
    # ব্যবহারকারীর প্রশ্ন প্রদর্শন
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # সেশনে সংরক্ষণ
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # প্রশ্ন RAG সিস্টেমে পাঠানো
    with st.chat_message("assistant"):
        if st.session_state.active_rag is None:
            error_msg = "দয়া করে প্রথমে একটি PDF আপলোড করুন বা সংরক্ষিত মডেল লোড করুন।"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("উত্তর তৈরি হচ্ছে..."):
                try:
                    answer = st.session_state.active_rag.query(
                        question=prompt,
                        force_llm=False,
                        k=5
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"ত্রুটি: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# নীচের তথ্য
st.sidebar.info("""
**ব্যবহার নির্দেশিকা:**
1. প্রথমে সাইডবার থেকে একটি PDF আপলোড করুন
2. অথবা সংরক্ষিত মডেল লোড করুন
3. প্রশ্ন জিজ্ঞাসা করুন এবং উত্তর পান
""")