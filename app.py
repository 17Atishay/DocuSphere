# app.py
# Main Streamlit UI for DocuSphere
# Dual-mode RAG Intelligence: Document Mode + Research Mode
# Powered by Endee Vector Database + Gemini LLM

import streamlit as st
import tempfile
import os
import time
from endee_client import create_index, insert_vectors, query_index, list_indexes
from embedder import get_embeddings, get_single_embedding
from document_processor import process_document
from web_researcher import research_topic
from llm_handler import get_answer, get_summary

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuSphere",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F8EF7, #A259FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .mode-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    .status-success {
        color: #4CAF50;
        font-weight: 600;
    }
    .status-error {
        color: #f44336;
        font-weight: 600;
    }
    .chat-message-user {
        background: #1e1e2e;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #4F8EF7;
    }
    .chat-message-ai {
        background: #12121f;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #A259FF;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_index" not in st.session_state:
    st.session_state.current_index = None

if "current_mode" not in st.session_state:
    st.session_state.current_mode = None

if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False

if "current_source" not in st.session_state:
    st.session_state.current_source = None

if "raw_content" not in st.session_state:
    st.session_state.raw_content = None

if "summary" not in st.session_state:
    st.session_state.summary = None


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def process_and_store(chunks, metadata, index_name):
    """Embed chunks and store in Endee."""
    with st.spinner("🔢 Generating embeddings..."):
        vectors = get_embeddings(chunks)

    with st.spinner("📦 Storing vectors in Endee..."):
        create_index(index_name, dimension=384)
        success = insert_vectors(index_name, vectors, metadata)

    return success


def query_and_answer(question, index_name, mode):
    """Embed question, query Endee, get Gemini answer."""
    question_vector = get_single_embedding(question)
    context_chunks = query_index(index_name, question_vector, top_k=5)

    if not context_chunks:
        return "I couldn't find relevant information to answer your question."

    answer = get_answer(question, context_chunks, mode)
    return answer


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🌐 DocuSphere</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a document or enter a topic — DocuSphere researches and lets you converse with knowledge.</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## ⚙️ DocuSphere Settings")
    st.divider()

    # Mode Selection
    st.markdown("### 🔀 Select Mode")
    mode = st.radio(
        "Choose how you want to build your knowledge base:",
        ["📄 Document Mode", "🔍 Research Mode"],
        help="Document Mode: Upload PDF/DOCX | Research Mode: Enter any topic"
    )

    st.divider()

    # Endee Status
    st.markdown("### 🗄️ Endee Vector DB Status")
    @st.cache_data(ttl=5)
    def get_indexes():
        return list_indexes()
    try:
        indexes = list_indexes()
        st.markdown('<p class="status-success">● Connected</p>', unsafe_allow_html=True)
        if indexes:
            st.markdown(f"**Active Indexes:** {len(indexes)}")
            for idx in indexes:
                st.markdown(f"  - `{idx}`")
        else:
            st.markdown("No indexes yet.")
    except Exception:
        st.markdown('<p class="status-error">● Disconnected</p>', unsafe_allow_html=True)
        st.warning("Make sure Endee Docker container is running.")

    st.divider()

    # Clear Chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Reset Knowledge Base
    if st.button("🔄 Reset Knowledge Base"):
        st.session_state.knowledge_base_ready = False
        st.session_state.current_index = None
        st.session_state.current_source = None
        st.session_state.raw_content = None
        st.session_state.chat_history = []
        st.session_state.summary = None    # Add this line
        st.rerun()

# ─────────────────────────────────────────────
# MAIN CONTENT AREA
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    # ── DOCUMENT MODE ──
    if "Document" in mode:
        st.markdown("### 📄 Document Mode")
        st.markdown("Upload a PDF or Word document.")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx"],
            help="Supported formats: PDF, DOCX | Max size: 10MB"
        )

        # File size check (10MB limit)
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 10:
                st.error(f"❌ File too large ({file_size_mb:.1f}MB). Maximum allowed size is 10MB.")
                st.stop()
            else:
                st.success(f"✅ File received: `{uploaded_file.name}` ({file_size_mb:.1f}MB)")

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("🚀 Process & Store in Endee", use_container_width=True):
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f".{uploaded_file.name.split('.')[-1]}"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    with st.spinner("📖 Extracting text from document..."):
                        result = process_document(tmp_path)

                    if result:
                        index_name = uploaded_file.name.replace(".", "_").replace(" ", "_").lower()
                        success = process_and_store(
                            result["chunks"],
                            result["metadata"],
                            index_name
                        )

                        if success:
                            st.session_state.knowledge_base_ready = True
                            st.session_state.current_index = index_name
                            st.session_state.current_mode = "document"
                            st.session_state.current_source = uploaded_file.name
                            st.session_state.raw_content = " ".join(result["chunks"])
                            st.success(f"✅ Knowledge base ready! {len(result['chunks'])} chunks stored in Endee.")
                        else:
                            st.error("❌ Failed to store in Endee. Check if Docker is running.")
                    else:
                        st.error("❌ Could not extract text from document.")

                    os.unlink(tmp_path)

            with col_b:
                if st.session_state.raw_content and st.button("📝 Summarize Document", use_container_width=True):
                    with st.spinner("✍️ Generating summary..."):
                        summary = get_summary(st.session_state.raw_content)
                    st.session_state.summary = summary

                if "summary" in st.session_state and st.session_state.summary:
                    with st.expander("📋 Document Summary", expanded=True):
                        st.markdown(st.session_state.summary)

    # ── RESEARCH MODE ──
    elif "Research" in mode:
        st.markdown("### 🔍 Research Mode")
        st.markdown("Enter any topic and DocuSphere will research it from the web.")

        topic = st.text_input(
            "Enter a topic to research:",
            placeholder="e.g. Quantum Computing, Climate Change, LLMs...",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🌐 Research & Store in Endee", use_container_width=True):
                if not topic.strip():
                    st.warning("Please enter a topic first.")
                else:
                    with st.spinner(f"🔍 Researching '{topic}'..."):
                        result = research_topic(topic)

                    if result:
                        index_name = topic.strip().replace(" ", "_").lower()[:30]
                        success = process_and_store(
                            result["chunks"],
                            result["metadata"],
                            index_name
                        )

                        if success:
                            st.session_state.knowledge_base_ready = True
                            st.session_state.current_index = index_name
                            st.session_state.current_mode = "research"
                            st.session_state.current_source = topic
                            st.session_state.raw_content = " ".join(result["chunks"])
                            st.success(f"✅ Research complete! {len(result['chunks'])} chunks stored in Endee.")
                        else:
                            st.error("❌ Failed to store in Endee.")
                    else:
                        st.error("❌ No content found for this topic.")

        with col_b:
            if st.session_state.raw_content and st.button("📝 Summarize Research", use_container_width=True):
                    with st.spinner("✍️ Generating summary..."):
                        summary = get_summary(st.session_state.raw_content)
                    st.session_state.summary = summary

            if "summary" in st.session_state and st.session_state.summary:
                with st.expander("📋 Research Summary", expanded=True):
                    st.markdown(st.session_state.summary)

with col2:
    # ── CHAT INTERFACE ──
    st.markdown("### 💬 Chat with Your Knowledge Base")

    if not st.session_state.knowledge_base_ready:
        st.info("👈 Process a document or research a topic first to start chatting.")
    else:
        source_label = st.session_state.current_source
        mode_label = "📄 Document" if st.session_state.current_mode == "document" else "🔍 Research"
        st.success(f"{mode_label} | Active: `{source_label}`")

        # Display chat history
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message-user">🧑 <b>You:</b> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message-ai">🤖 <b>DocuSphere:</b> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )

        # Question input
        question = st.chat_input("Ask anything about your knowledge base...")

        if question:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            # Get answer
            with st.spinner("🤔 Thinking..."):
                answer = query_and_answer(
                    question,
                    st.session_state.current_index,
                    st.session_state.current_mode
                )

            # Add AI response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><sub>Built by Atishay Jain | Powered by Endee Vector DB + Gemini + HuggingFace</sub></center>",
    unsafe_allow_html=True
)
