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
        "qa_agent": None,
        "selected_model": "gemini-pro"
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Model Configuration
MODEL_OPTIONS = {
    "gemini-pro": {
        "name": "Google Gemini Pro",
        "key_env": "GOOGLE_API_KEY",
        "class": ChatGoogleGenerativeAI
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "key_env": "DEEPSEEK_API_KEY",
        "class": None  # Replace with actual DeepSeek class
    }
}

# Custom CSS styling
st.markdown("""
<style>
    .header { 
        padding: 20px;
        background: linear-gradient(45deg, #2E86C1, #3498DB);
        color: white;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    }
    .model-selector {
        margin-bottom: 1.5rem;
        padding: 10px;
        border-radius: 8px;
        background: #f8f9fa;
    }
    .upload-section { 
        border: 2px dashed #2E86C1;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-bubble { 
        padding: 15px 20px;
        margin: 12px 0;
        max-width: 80%;
        clear: both;
        border-radius: 15px;
    }
    .user { 
        background: #2E86C1;
        color: white;
        float: right;
    }
    .assistant { 
        background: #f0f2f6;
        color: #2c3e50;
        float: left;
    }
    .footer { 
        margin-top: 50px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📄 Multi-Model Doc Analyzer</h1></div>', unsafe_allow_html=True)

# Model Selection
with st.container():
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS[x]["name"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# API Key Handling
try:
    api_key = st.secrets[MODEL_OPTIONS[selected_model]["key_env"].lower()]
except:
    api_key = os.getenv(MODEL_OPTIONS[selected_model]["key_env"])

if not api_key:
    st.error(f"{MODEL_OPTIONS[selected_model]['name']} API key not found!")
    st.stop()

def handle_file_upload():
    with st.container():
        uploaded_file = st.file_uploader(
            "📤 Upload PDF Document",
            type=["pdf"],
            key=f"uploader_{st.session_state.file_key}"
        )

        if uploaded_file:
            if st.session_state.processed_docs and st.session_state.processed_docs.get('file_id') == uploaded_file.file_id:
                st.info("Document already processed")
                return True

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                
                with st.spinner("Analyzing document..."):
                    loader = PyPDFLoader(tmp_file.name)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=10000,
                        chunk_overlap=1000
                    )
                    chunks = text_splitter.split_documents(documents)

                    st.session_state.processed_docs = {
                        'chunks': chunks[:4],
                        'file_id': uploaded_file.file_id
                    }
                    os.unlink(tmp_file.name)
                    return True

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.file_key += 1
                return False

def initialize_agent():
    if not st.session_state.qa_agent or st.session_state.selected_model != selected_model:
        try:
            model_config = MODEL_OPTIONS[selected_model]
            
            if selected_model == "gemini-pro":
                llm = model_config["class"](
                    model=selected_model,
                    temperature=0.3,
                    google_api_key=api_key
                )
            elif selected_model == "deepseek-r1":
                # Replace with actual DeepSeek initialization
                llm = model_config["class"](
                    model_name=selected_model,
                    api_key=api_key
                )
            
            st.session_state.qa_agent = LLMChain(
                llm=llm,
                prompt=PromptTemplate.from_template("""
                Analyze the document and provide a concise answer:
                
                Context: {context}
                Question: {question}
                
                Answer format:
                - Direct answer (1-2 sentences)
                - 3 key supporting points
                """)
            )
            st.session_state.selected_model = selected_model
            
        except Exception as e:
            st.error(f"Model initialization failed: {str(e)}")
    return st.session_state.qa_agent

def process_question(question):
    try:
        qa_agent = initialize_agent()
        context = " ".join([
            chunk.page_content[:2000] 
            for chunk in st.session_state.processed_docs['chunks'][:3]
        )

        with st.spinner("Generating answer..."):
            response = qa_agent.run({
                "context": context,
                "question": question
            })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        st.rerun()

def main():
    if handle_file_upload():
        st.markdown("### 💬 Document Q&A")
        
        for message in st.session_state.messages:
            css_class = "user" if message["role"] == "user" else "assistant"
            st.markdown(
                f'<div class="chat-bubble {css_class}">{message["content"]}</div>',
                unsafe_allow_html=True
            )
        
        question = st.text_input("Ask your question:", key="question_input")
        
        if st.button("Get Answer") and question:
            st.session_state.messages.append({"role": "user", "content": question})
            process_question(question)
    
    st.markdown("""
    <div class="footer">
        <p>Developed by Waqas Baloch • Contact: waqaskhosa99@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
