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
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# API Key Handling (Google only)
try:
    GOOGLE_API_KEY = st.secrets["google"]["api_key"]
except KeyError:
    st.error("Google API key not found in secrets!")
    st.stop()

# Enhanced responsive CSS styling
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
.upload-section {
    border: 2px dashed #2E86C1;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-container {
    width: 100%;
    max-width: 800px;
    margin: 0 auto;
}
.chat-bubble {
    padding: 15px 20px;
    margin: 12px 0;
    width: fit-content;
    max-width: 80%;
    border-radius: 15px;
    display: flex;
    align-items: flex-start;
    clear: both;
}
.user {
    background: #2E86C1;
    color: white;
    margin-left: auto;
}
.assistant {
    background: #f0f2f6;
    color: #2c3e50;
    margin-right: auto;
}
.chat-bubble .icon {
    margin: 0 10px;
    font-size: 1.2em;
}
.footer {
    margin-top: 50px;
    padding: 20px;
    text-align: center;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .chat-bubble {
        max-width: 90%;
        padding: 12px 16px;
        font-size: 14px;
    }
    .chat-bubble .icon {
        font-size: 1em;
        margin: 0 8px;
    }
    .upload-section {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📄 Doc Analyzer</h1></div>', unsafe_allow_html=True)

def handle_file_upload():
    with st.container():
        uploaded_file = st.file_uploader(
            "📤 Upload PDF Document", type=["pdf"], key=f"uploader_{st.session_state.file_key}"
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

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    chunks = text_splitter.split_documents(documents)

                    st.session_state.processed_docs = {'chunks': chunks[:4], 'file_id': uploaded_file.file_id}
                    os.unlink(tmp_file.name)
                    return True

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.file_key += 1
                return False
        return False


def initialize_agent():
    if not st.session_state.qa_agent:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY
            )

            st.session_state.qa_agent = LLMChain(
                llm=llm,
                prompt=PromptTemplate.from_template("""
                Analyze the document and provide a concise answer:

                Context: {context}
                Question: {question}

                Answer format:
                - Analyze the document and provide a concise answer
               
                """)
            )

        except Exception as e:
            st.error(f"Model initialization failed: {str(e)}")
    return st.session_state.qa_agent


def process_question(question):
    try:
        qa_agent = initialize_agent()
        context = " ".join([chunk.page_content[:2000] for chunk in st.session_state.processed_docs['chunks'][:3]])

        with st.spinner("Generating answer..."):
            response = qa_agent.run({"context": context, "question": question})

        if response:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        st.rerun()

def main():
    if handle_file_upload():
        st.markdown("### 💬 Document Q&A")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for message in st.session_state.messages:
            css_class = "user" if message["role"] == "user" else "assistant"
            icon = "👤" if message["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="chat-bubble {css_class}"><span class="icon">{icon}</span>{message["content"]}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)
        question = st.text_input("Ask your question:", key="question_input")

        if st.button("Get Answer") and question:
            process_question(question)

    st.markdown("""
    <div class="footer">
        <p>Developed by Waqas Baloch • Contact: waqaskhosa99@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
