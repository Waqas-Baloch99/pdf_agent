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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = None
    if "file_key" not in st.session_state:
        st.session_state.file_key = 0

initialize_session_state()

# Google API Key
GOOGLE_API_KEY = st.secrets["google"]["api_key"]
if not GOOGLE_API_KEY:
    st.error("Google API key not found! Please configure it in your environment.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .header { padding: 20px; background: linear-gradient(45deg, #2E86C1, #3498DB); color: white;
              border-radius: 15px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .upload-section { border: 2px dashed #2E86C1; border-radius: 10px; padding: 2rem; text-align: center; margin-bottom: 2rem;}
    .chat-user { background: #2E86C1; color: white; padding: 12px 18px; border-radius: 20px 20px 0 20px;
                 margin: 8px 0; max-width: 80%; float: right; clear: both;}
    .chat-ai { background: #F8F9FA; color: #2C3E50; padding: 12px 18px; border-radius: 20px 20px 20px 0;
               margin: 8px 0; max-width: 80%; float: left; clear: both; border: 1px solid #DEE2E6;}
    .footer { margin-top: 50px; padding: 20px; background: #2C3E50; color: white; text-align: center; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📑 SmartDoc Analyzer Pro</h1></div>', unsafe_allow_html=True)

# Handle File Upload
def handle_file_upload():
    """Upload and process a PDF file"""
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag and drop or click to upload PDF", type=["pdf"],
                                         key=f"uploader_{st.session_state.file_key}", help="Max file size: 50MB")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Prevent reprocessing the same file
            if st.session_state.processed_docs and st.session_state.processed_docs.get('file_id') == uploaded_file.file_id:
                st.info("This document has already been processed.")
                return True

            with st.spinner("Analyzing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                    
                    loader = PyPDFLoader(tmp_file.name)
                    documents = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    chunks = text_splitter.split_documents(documents)

                    st.session_state.processed_docs = {'chunks': chunks, 'file_id': uploaded_file.file_id}
                    st.session_state.messages = []
                    return True
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    return False

# Initialize LLM Agent
def initialize_agent():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_output_tokens=2048)
        prompt_template = """**Document Analysis Task**
        You are an expert document analyst. Follow these guidelines:
        1. Answer strictly based on the context
        2. Acknowledge uncertainty when needed
        3. Format answers with markdown
        4. Highlight key points in **bold**
        5. Keep answers under 300 words

        **Context:**
        {context}

        **Question:** {question}

        **Analysis Report:**"""
        
        return LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    
    except Exception as e:
        st.error(f"Failed to initialize AI agent: {str(e)}")
        return None

# Process Questions
def process_question(question):
    """Handle user questions about the document"""
    st.session_state.messages.append({"role": "user", "content": question})
    
    try:
        qa_agent = initialize_agent()
        if not qa_agent:
            return

        context = "\n".join([chunk.page_content for chunk in st.session_state.processed_docs['chunks'][:4]])

        with st.spinner("Analyzing content..."):
            response = qa_agent.run({"context": context, "question": question})

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# Chat Interface
def chat_interface():
    """Chat interface with document preview and Q&A"""
    if st.session_state.processed_docs:
        with st.expander("📄 Document Preview", expanded=True):
            preview_text = "\n".join([doc.page_content for doc in st.session_state.processed_docs['chunks'][:2]])
            st.markdown(f"```\n{preview_text[:500]}\n...```")
        
        st.markdown("### 💬 Document Q&A")
        
        # Display chat history
        for message in st.session_state.messages:
            role, content = message["role"], message["content"]
            chat_class = "chat-user" if role == "user" else "chat-ai"
            icon = "👤" if role == "user" else "🤖"
            st.markdown(f'<div class="{chat_class}">{icon} {content}</div>', unsafe_allow_html=True)
        
        # User Input
        question = st.text_input("Ask about the document:", placeholder="Type your question here...", key="question_input")
        
        if st.button("🚀 Ask", use_container_width=True) and question:
            process_question(question)

# Main Function
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
