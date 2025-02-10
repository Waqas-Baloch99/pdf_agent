import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from deepseek_ai import DeepSeekAI  # Import DeepSeek library

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
    # ... (Your file upload code - same as before)
    pass # Placeholder for your existing code

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
            elif st.session_state.selected_model == "deepseek-r1": # DeepSeek model
                deepseek = DeepSeekAI(api_key=DEEPSEEK_API_KEY)  # Initialize DeepSeek client
                st.session_state.qa_agent = LLMChain(
                    llm=deepseek,  # Use DeepSeek LLM
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
    # ... (Your question processing code - same as before)
    pass # Placeholder for your existing code

def chat_interface():
    # ... (Your chat interface code - same as before)
    pass # Placeholder for your existing code


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
