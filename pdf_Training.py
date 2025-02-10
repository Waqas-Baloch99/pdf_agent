import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Load environment variables
load_dotenv()

# Custom CSS for styling
st.markdown("""
<style>
    .header { 
        padding: 20px;
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        text-align: center;
    }
    .chat-user {
        background-color: #2E86C1;
        color: white;
        padding: 10px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .chat-ai {
        background-color: #EBEDEF;
        color: #2C3E50;
        padding: 10px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #2C3E50;
        color: white;
        text-align: center;
        padding: 10px;
        left: 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for API key
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables!")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# App Header
st.markdown('<div class="header"><h1>📑 SmartDoc Analyzer</h1></div>', unsafe_allow_html=True)
st.markdown("---")

def process_pdf(file):
    """Process PDF file with error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
        
        loader = PyPDFLoader(tmp_file.name)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        return text_splitter.split_documents(documents), tmp_file.name
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None
    finally:
        if 'tmp_file' in locals() and tmp_file.name:
            os.unlink(tmp_file.name)

def initialize_agent():
    """Initialize the Q&A agent"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        max_output_tokens=2048
    )

    template = """You are a professional document analyst. Answer based on the context.
    If answer isn't in document, state that clearly. Be concise and accurate.

    Context:
    {context}

    Question: {question}
    Analytical Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    return LLMChain(llm=llm, prompt=prompt)

# Initialize agent
qa_agent = initialize_agent()

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload PDF Document", type=["pdf"], 
                                help="Max file size: 50MB")

# Reset chat history if new file uploaded
if uploaded_file and uploaded_file != st.session_state.current_file:
    st.session_state.messages = []
    st.session_state.current_file = uploaded_file

if uploaded_file:
    with st.spinner("🔍 Analyzing document..."):
        chunks, tmp_path = process_pdf(uploaded_file)
    
    if chunks:
        # Document preview
        with st.expander("📄 Document Preview"):
            preview_text = "\n".join([doc.page_content for doc in chunks[:2]])
            st.markdown(f"```\n{preview_text[:1000]}\n...```")
        
        # Chat interface
        st.markdown("### 💬 Document Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                st.markdown(f'<div class="chat-user">👤 User: {content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 Analyst: {content}</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input("Ask about the document:", 
                               placeholder="Type your question here...",
                               key="question_input")
        
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("🚀 Ask"):
                if not question:
                    st.warning("Please enter a question")
                else:
                    # Add user question to history
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # Get context
                    context = "\n".join([chunk.page_content for chunk in chunks[:4]])
                    
                    with st.spinner("💡 Analyzing..."):
                        try:
                            response = qa_agent.run({
                                "context": context,
                                "question": question
                            })
                            
                            # Add AI response to history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Rerun to update chat display
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Waqas Baloch • 📧 <a href="mailto:waqaskhosa99@gmail.com" style="color: white;">waqaskhosa99@gmail.com</a></p>
</div>
""", unsafe_allow_html=True)
