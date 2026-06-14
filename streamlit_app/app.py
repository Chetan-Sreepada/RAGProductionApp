import streamlit as st
import requests
import json
from typing import Dict, Any
import PyPDF2
from io import BytesIO
import docx
import os

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple clean CSS
st.markdown("""
    <style>
    /* Clean simple styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        color: #000;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        color: #000;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Status indicators */
    .status-connected {
        background-color: #4CAF50;
        padding: 8px;
        border-radius: 5px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    .status-disconnected {
        background-color: #f44336;
        padding: 8px;
        border-radius: 5px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    /* Remove extra spacing */
    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def check_backend_health() -> bool:
    """Check if backend is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def ingest_document(text: str, doc_id: str = None) -> Dict[str, Any]:
    """Send document to ingestion endpoint"""
    try:
        payload = {"text": text}
        if doc_id:
            payload["document_id"] = doc_id
        
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def send_chat_message(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Send chat message to backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("📚 Document Manager")
    
    # Status
    if check_backend_health():
        st.markdown('<div class="status-connected">✅ Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-disconnected">❌ Backend Disconnected</div>', unsafe_allow_html=True)
        st.warning("Start backend: python -m uvicorn main:app --reload")
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📄 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_file is not None:
        # Extract text based on file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            if file_extension == 'pdf':
                text = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                text = extract_text_from_docx(uploaded_file)
            elif file_extension == 'txt':
                text = extract_text_from_txt(uploaded_file)
            else:
                text = ""
                st.error("Unsupported file type")
            
            if text.strip():
                # Ingest the extracted text
                doc_id = uploaded_file.name.replace('.', '_')
                result = ingest_document(text, doc_id)
                
                if result["success"]:
                    st.success(f"✅ Successfully ingested {uploaded_file.name}")
                    st.info(f"📊 Created {result['data']['chunks_stored']} chunks")
                else:
                    st.error(f"❌ Failed: {result['error']}")
            else:
                st.error("No text could be extracted from the file")
    
    # Manual text input
    with st.expander("✏️ Or paste text manually"):
        manual_text = st.text_area(
            "Paste your text here:",
            height=150,
            placeholder="Paste your document content here..."
        )
        
        if st.button("📥 Ingest Text", use_container_width=True):
            if manual_text.strip():
                with st.spinner("Ingesting..."):
                    result = ingest_document(manual_text)
                    if result["success"]:
                        st.success(f"✅ Ingested {result['data']['chunks_stored']} chunks!")
                    else:
                        st.error(f"Failed: {result['error']}")
            else:
                st.warning("Please enter some text")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    default_top_k = st.slider("Context chunks", 1, 7, 3, help="Number of text chunks to retrieve")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tips:**")
    st.caption("• Upload PDF, DOCX, or TXT files")
    st.caption("• Ask questions about your documents")
    st.caption("• Data persists after restart")

# ==================== MAIN CHAT INTERFACE ====================
# Simple header
st.title("💬 RAG Chat Assistant")
st.caption("Ask questions about your uploaded documents")

# Chat history display
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("👋 No messages yet. Upload a document and start asking questions!")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_query = st.text_input(
        "Ask a question:",
        placeholder="Type your question here...",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", use_container_width=True)

# Handle sending
if send_button and user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Check backend
    if not check_backend_health():
        error_msg = "❌ Backend not connected. Please start the backend server."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.rerun()
    
    # Get response
    with st.spinner("🤔 Thinking..."):
        result = send_chat_message(user_query, default_top_k)
    
    if result["success"]:
        answer = result["data"]["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Show context if available (optional - in expander)
        if result["data"].get("retrieved_contexts"):
            with st.expander("📚 View source context", expanded=False):
                for i, context in enumerate(result["data"]["retrieved_contexts"], 1):
                    st.text(f"Context {i}:")
                    st.text(context[:300] + "..." if len(context) > 300 else context)
                    st.markdown("---")
    else:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {result['error']}"})
    
    st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by FAISS | Sentence Transformers | Google Gemini")