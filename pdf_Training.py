import os
import streamlit as st
from dotenv import load_dotenv
import requests
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
        "selected_model": "gemini-pro"  # Default model
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# API Key Configuration (Google and DeepSeek)
try:
    GOOGLE_API_KEY = st.secrets["google"]["api_key"]
    DEEPSEEK_API_KEY = st.secrets["deepseek"]["api_key"]  # Get DeepSeek key
except (KeyError, AttributeError):
    st.error("API keys not found in secrets!")
    st.stop()

# DeepSeek API Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/inference"  # Replace with actual DeepSeek API URL

# Custom CSS (same as before)
st.markdown("""
<style>
    /* ... (Your CSS code here) ... */
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📑 SmartDoc Analyzer Pro</h1></div>', unsafe_allow_html=True)

# Model Selection
st.markdown("### Select Language Model:")
model_options = ["gemini-pro", "deepseek-r1"]
st.session_state.selected_model = st.selectbox("Choose a model:", model_options, index=model_options.index(st.session_state.selected_model))

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
                        'chunks': chunks[:4],  # Limit to first 4 chunks
                        'file_id': uploaded_file.file_id
                    }
                    st.session_state.messages = []
                    os.unlink(tmp_file.name)
                    return True

            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.session_state.file_key += 1
                return False
        return False #Return false if no file is uploaded

def initialize_agent():
    if not st.session_state.qa_agent:
        try:
            if st.session_state.selected_model == "gemini-pro":
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
            elif st.session_state.selected_model == "deepseek-r1":
                st.session_state.qa_agent = DeepSeekAPIAgent(DEEPSEEK_API_KEY, DEEPSEEK_API_URL)

        except Exception as e:
            st.error(f"🤖 Agent initialization failed: {str(e)}")
    return st.session_state.qa_agent

class DeepSeekAPIAgent:
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url

    def run(self, inputs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": f"{inputs['context']}\n{inputs['question']}",
            # Add other DeepSeek API parameters as needed (e.g., temperature, max_tokens)
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10) # Set a timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            answer = self.extract_answer(result)
            return answer
        except requests.exceptions.RequestException as e:
            st.error(f"DeepSeek API Error: {e}")
            return None
        except Exception as e: # Catch any other exceptions
            st.error(f"An unexpected error occurred: {e}")
            return None

    def extract_answer(self, result):
        try:
            # Adapt this to the actual structure of DeepSeek's API response
            # Example (you'll likely need to change this):
            return result.get('choices', [{}])[0].get('text', "Could not extract answer")  # Handle missing keys
        except (KeyError, IndexError, TypeError) as e:  # Handle potential errors
            st.error(f"Error extracting answer: {e}.  Check DeepSeek API response format.")
            return "Could not extract answer"


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

        if response:  # Check if response is not None
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

    except Exception as e:
        st.error(f"⚠️ Analysis error: {str(e)}")
    finally:
        st.rerun()  # Rerun to update the chat display


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
