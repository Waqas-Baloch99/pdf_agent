import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Load environment variables
load_dotenv()

# Initialize session state
def initialize_session_state():
    session_defaults = {
        "messages": [],
        "processed_docs": None,
        "file_key": 0,
        "qa_agent": None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Google API Key Configuration
GOOGLE_API_KEY = st.secrets["google"]["api_key"]
if not GOOGLE_API_KEY:
    st.error("Google API key not found! Configure it in .env or secrets.")
    st.stop()

# Custom CSS for improved performance
st.markdown("""
<style>
    .header { 
        padding: 20px; 
        background: linear-gradient(45deg, #2E86C1, #3498DB);
        color: white; border-radius: 15px;
        text-align: center; margin-bottom: 25px;
    }
    .upload-section { 
        border: 2px dashed #2E86C1;
        border-radius: 10px; padding: 2rem;
        text-align: center; margin-bottom: 2rem;
    }
    .chat-bubble { 
        padding: 12px 18px; margin: 8px 0;
        max-width: 80%; clear: both;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user { 
        background: #2E86C1; color: white;
        border-radius: 20px 20px 0 20px; float: right;
    }
    .assistant { 
        background: #F8F9FA; color: #2C3E50;
        border-radius: 20px 20px 20px 0; float: left;
        border: 1px solid #DEE2E6;
    }
    .footer { 
        margin-top: 50px; padding: 20px;
        background: #2C3E50; color: white;
        text-align: center; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📑 SmartDoc Analyzer Pro</h1></div>', unsafe_allow_html=True)

def handle_file_upload():
    """Handle PDF upload and processing with caching"""
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📤 Upload PDF Document (Max 50MB)",
            type=["pdf"],
            key=f"uploader_{st.session_state.file_key}",
            help="Supported formats: PDF"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            if (st.session_state.processed_docs and 
                st.session_state.processed_docs.get('file_id') == uploaded_file.file_id):
                st.info("✅ Document already processed")
                return True

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                
                with st.spinner("🔍 Analyzing document structure..."):
                    loader = PyPDFLoader(tmp_file.name)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=8000,  # Reduced chunk size for faster processing
                        chunk_overlap=800
                    )
                    chunks = text_splitter.split_documents(documents)

                    st.session_state.processed_docs = {
                        'chunks': chunks[:6],  # Use only first 6 chunks
                        'file_id': uploaded_file.file_id
                    }
                    st.session_state.messages = []
                    os.unlink(tmp_file.name)
                    return True

            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.session_state.file_key += 1
                return False

def initialize_agent():
    """Initialize and cache the LLM agent"""
    if not st.session_state.qa_agent:
        try:
            st.session_state.qa_agent = LLMChain(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0.3,
                    max_output_tokens=1024,  # Reduced for faster responses
                    google_api_key=GOOGLE_API_KEY
                ),
                prompt=PromptTemplate.from_template("""
                **Document Analysis Task**
                Context: {context}
                Question: {question}
                
                Respond concisely in this format:
                📌 **Key Points**: [3 bullet points]
                🎯 **Direct Answer**: [Clear response under 100 words]
                """)
            )
        except Exception as e:
            st.error(f"🤖 Agent initialization failed: {str(e)}")
    return st.session_state.qa_agent

def process_question(question):
    """Handle question processing with streaming"""
    try:
        qa_agent = initialize_agent()
        if not qa_agent:
            return

        context = " ".join([
            chunk.page_content[:2000]  # Truncate long content
            for chunk in st.session_state.processed_docs['chunks'][:3]  # Use first 3 chunks
        ])

        with st.spinner("💡 Analyzing..."):
            response = qa_agent.run({
                "context": context,
                "question": question
            })

        # Add response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
    except Exception as e:
        st.error(f"⚠️ Analysis error: {str(e)}")
    finally:
        st.rerun()

def chat_interface():
    """Interactive chat interface"""
    if st.session_state.processed_docs:
        with st.expander("📄 Document Preview", expanded=False):
            preview_text = " ".join([
                doc.page_content[:500] 
                for doc in st.session_state.processed_docs['chunks'][:2]
            )
            st.markdown(f"```\n{preview_text}\n...```")
        
        st.markdown("### 💬 Document Q&A")
        
        # Display chat history
        for message in st.session_state.messages:
            css_class = "user" if message["role"] == "user" else "assistant"
            icon = "👤" if message["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="chat-bubble {css_class}">{icon} {message["content"]}</div>',
                unsafe_allow_html=True
            )
        
        # Question input
        question = st.text_input(
            "Ask about the document:", 
            placeholder="Type your question...",
            key="question_input",
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Ask", use_container_width=True) and question:
            st.session_state.messages.append({"role": "user", "content": question})
            process_question(question)

def main():
    if handle_file_upload():
        chat_interface()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed by Waqas Baloch • 📧 <a href="mailto:waqaskhosa99@gmail.com" style="color: white;">waqaskhosa99@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
