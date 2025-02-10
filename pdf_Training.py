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
try:
    GOOGLE_API_KEY = st.secrets["google"]["api_key"]
except (KeyError, AttributeError):
    st.error("Google API key not found in secrets!")
    st.stop()

# Custom CSS with animations and styling
st.markdown("""
<style>
    .header { 
        padding: 20px;
        background: linear-gradient(45deg, #2E86C1, #3498DB);
        color: white;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .upload-section { 
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        background: #f8f9fa;
    }
    .chat-bubble { 
        padding: 15px 20px;
        margin: 12px 0;
        max-width: 80%;
        clear: both;
        position: relative;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .user { 
        background: #2E86C1;
        color: white;
        border-radius: 18px 18px 0 18px;
        float: right;
        animation: slideInRight 0.3s ease;
    }
    .assistant { 
        background: #ffffff;
        color: #2c3e50;
        border-radius: 18px 18px 18px 0;
        float: left;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: typing 1s steps(40), blinkCaret 0.75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blinkCaret {
        from, to { border-color: transparent }
        50% { border-color: #2E86C1 }
    }
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0 }
        to { transform: translateX(0); opacity: 1 }
    }
    .footer { 
        margin-top: 50px;
        padding: 20px;
        color: #666;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📑 SmartDoc Analyzer Pro</h1></div>', unsafe_allow_html=True)

def handle_file_upload():
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📤 Upload PDF Document (Max 50MB)",
            type=["pdf"],
            key=f"uploader_{st.session_state.file_key}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            if st.session_state.processed_docs and st.session_state.processed_docs.get('file_id') == uploaded_file.file_id:
                st.info("✅ Document already processed")
                return True

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                
                with st.spinner("🔍 Analyzing document..."):
                    loader = PyPDFLoader(tmp_file.name)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=6000,
                        chunk_overlap=600
                    )
                    chunks = text_splitter.split_documents(documents)

                    st.session_state.processed_docs = {
                        'chunks': chunks[:4],
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
    if not st.session_state.qa_agent:
        try:
            st.session_state.qa_agent = LLMChain(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0.3,
                    max_output_tokens=800,
                    google_api_key=GOOGLE_API_KEY
                ),
                prompt=PromptTemplate.from_template("""
                Provide concise answer based on this document content:
                
                Context: {context}
                Question: {question}
                
                Answer:
                [Provide a relevant and short answer based on the document content.]
                """)
            )
        except Exception as e:
            st.error(f"🤖 Agent initialization failed: {str(e)}")
    return st.session_state.qa_agent

def process_question(question):
    try:
        qa_agent = initialize_agent()
        if not qa_agent:
            return

        context = " ".join([
            chunk.page_content[:2000]
            for chunk in st.session_state.processed_docs['chunks'][:3]
        ])

        with st.spinner("💡 Analyzing..."):
            response = qa_agent.run({
                "context": context,
                "question": question
            })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
    except Exception as e:
        st.error(f"⚠️ Analysis error: {str(e)}")
    finally:
        st.rerun()

def chat_interface():
    if st.session_state.processed_docs:
        with st.expander("📄 Document Preview", expanded=False):
            preview_text = " [...] ".join([
                doc.page_content[:400] 
                for doc in st.session_state.processed_docs['chunks'][:2]
            ])
            st.markdown(f"```\n{preview_text}\n...```")
        
        st.markdown("### 💬 Document Q&A")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble user">👤 {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble assistant">🤖 {message["content"]}</div>',
                    unsafe_allow_html=True
                )
        
        question = st.text_input(
            "Ask your question:",
            placeholder="Type question here...",
            key="question_input",
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Get Answer", use_container_width=True) and question:
            st.session_state.messages.append({"role": "user", "content": question})
            process_question(question)

def main():
    if handle_file_upload():
        chat_interface()
    
    st.markdown("""
    <div class="footer">
        <p>Developed by Waqas Baloch • <a href="mailto:waqaskhosa99@gmail.com">Contact</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
