import os
import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import base64
import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document

# --------- Custom Beast Mode Styling ---------
st.set_page_config(
    page_title="🦾 RAG AI PDF BEAST",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)
custom_css = """
<style>
body {
    background: linear-gradient(130deg, #161a24 0%, #001858 100%) !important;
    color: #f1f3f6;
}
.sidebar .sidebar-content {
    background: #001858;
}
.stButton > button {
    background: #1e2746;
    color: #ffeedd;
    font-weight: bold;
    border-radius: 8px;
    transition: 0.2s;
}
.stButton > button:hover {
    background: #ff6b35;
    color: #fff;
}
h1, h2, h3 {
    color: #ffbe0b;
}
hr {
    border: none;
    border-top: 2px solid #ffbe0b;
}
div[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin: 8px 0;
    padding: 8px 12px;
    background: #fff;
    border-left: 4px solid #000000;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# ---------------------------------------------

# ---------- OCR + Poppler + Tesseract Config ---------
POPPLER_PATH = r"C:\Users\abhip\Downloads\poppler-24.08.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def ocr_pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    texts = []
    for img in images:
        text = pytesseract.image_to_string(img)
        texts.append(text)
    return texts

def ocr_text_to_documents(ocr_texts):
    return [Document(page_content=page_text, metadata={"page": i + 1}) for i, page_text in enumerate(ocr_texts)]

# ---------- Streamlit State Defaults ----------
for var in ["conversation", "chat_history", "document_summary", "suggested_questions", "processed_file"]:
    if var not in st.session_state:
        st.session_state[var] = [] if var == "chat_history" else None

# ---------- Env/Keys/LLM Initialization ----------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY") or "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"
os.environ["GROQ_API_KEY"] = API_KEY

def create_llm():
    return ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        groq_api_key=os.environ["GROQ_API_KEY"],
    )

# ---------- PDF Processing Pipeline ----------
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name

    with st.spinner("🦾 Crunching PDF... Summoning AI beast..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Fallback to OCR if no text extracted
        if not documents or all(not doc.page_content.strip() for doc in documents):
            st.info("No selectable text found. Attempting OCR (slower, but works for scanned/image PDFs)...")
            ocr_texts = ocr_pdf_to_text(pdf_path)
            try:
                os.unlink(pdf_path)
            except Exception:
                pass
            documents = ocr_text_to_documents(ocr_texts)
        else:
            os.unlink(pdf_path)

        if not documents or all(not doc.page_content.strip() for doc in documents):
            st.error("No text could be extracted from this PDF—even after OCR. The document may be encrypted or corrupted.")
            return None, None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        if not chunks:
            st.error("The PDF was loaded, but no valid text chunks were generated.")
            return None, None

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore, chunks

def generate_summary(chunks):
    llm = create_llm()
    combined_text = " ".join([chunk.page_content for chunk in chunks[:15]])
    summary_prompt = PromptTemplate.from_template(
        """You are an expert document analyst. 
        Based on the following document excerpts, generate a concise summary with 5-10 key points.

        DOCUMENT EXCERPTS:
        {text}

        SUMMARY (5-10 bullet points):"""
    )
    summary = llm.invoke(summary_prompt.format(text=combined_text))
    return summary.content

def generate_suggested_questions(chunks):
    llm = create_llm()
    combined_text = " ".join([chunk.page_content for chunk in chunks[:15]])
    question_prompt = PromptTemplate.from_template(
        """You are an expert document analyst. 
        Based on the following document excerpts, generate 5-7 insightful questions that a user might want to ask about this document.
        The questions should be varied and cover different aspects of the content.
        Format the questions as a numbered list.

        DOCUMENT EXCERPTS:
        {text}

        SUGGESTED QUESTIONS (5-7 questions):"""
    )
    questions = llm.invoke(question_prompt.format(text=combined_text))
    lines = [q.strip() for q in questions.content.split("\n") if q.strip()]
    questions_list = [q.split('. ', 1)[-1] if '. ' in q else q for q in lines]
    questions_list = [q.lstrip('1234567890.- ') for q in questions_list]
    return questions_list

def get_conversation_chain(vectorstore):
    llm = create_llm()
    qa_template = """
    You are an intelligent assistant that helps users understand documents. Answer the question based on the context provided.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history}

    QUESTION:
    {question}

    YOUR ANSWER:"""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "chat_history", "question"])
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

def handle_user_question(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload a PDF document first.")
        return
    with st.spinner("🤖 Thinking..."):
        response = st.session_state.conversation(
            {"question": user_question, "chat_history": st.session_state.chat_history}
        )
        answer = response["answer"]
        sources = response.get("source_documents", [])
        if sources:
            source_text = "\n\n**Sources:**\n"
            for i, source in enumerate(sources[:3]):
                page = source.metadata.get("page", "Unknown page")
                excerpt = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                source_text += f"\n**Page {page}:** {excerpt}\n"
            answer_with_sources = answer + source_text
        else:
            answer_with_sources = answer
        st.session_state.chat_history.append((user_question, answer_with_sources))
        st.rerun()
        return response

def get_download_link(chat_history, filename="chat_history.txt"):
    chat_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    b64 = base64.b64encode(chat_text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Chat History</a>'

# ----------------- Main App -----------------

st.title("🦾 RAG AI PDF BEAST")
st.markdown("""
> Welcome to the **ultimate** RAG PDF AI Chatbot!
>
> - 📝 Instant summaries & auto-generated smart starter questions
> - 🤖 Beast-level AI chat – ask anything about your doc, get page-sourced answers!
> - 🧠 All powered by Llama 3 70B + Hugging Face vector search
""")
st.divider()

with st.sidebar:
    st.header("📤 PDF Upload")
    uploaded_file = st.file_uploader("Drop your BEAST PDF here!", type="pdf")
    st.info("""
    - 💡 *Supports scanned/image PDFs (automatic OCR)*
    - 🏎️ *Processing time depends on size & image content*
    """)
    st.markdown("---")
    if st.session_state.chat_history:
        st.download_button(
            label="⏬ Download Chat History",
            data="\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]),
            file_name="chat_history.txt",
            mime="text/plain"
        )

if uploaded_file is not None and st.session_state.processed_file != uploaded_file.name:
    with st.spinner("🦾 Crunching PDF... Summoning AI beast..."):
        result = process_pdf(uploaded_file)
        if result is None or result[0] is None:
            st.session_state.conversation = None
            st.session_state.processed_file = None
        else:
            vectorstore, chunks = result
            st.session_state.conversation = get_conversation_chain(vectorstore)
            with st.spinner("Generating document summary..."):
                st.session_state.document_summary = generate_summary(chunks)
            with st.spinner("Generating suggested questions..."):
                st.session_state.suggested_questions = generate_suggested_questions(chunks)
            st.session_state.processed_file = uploaded_file.name
            st.success(f"✅ Document {uploaded_file.name} successfully processed! Ready for action!")
            st.balloons()
            st.session_state.chat_history = []

if uploaded_file is not None:
    tab1, tab2 = st.tabs([":star: Summary & Suggestions", ":robot_face: Chat with Document"])
    with tab1:
        st.header("📋 Document Summary")
        if st.session_state.document_summary:
            st.write(st.session_state.document_summary)
        else:
            st.info("Summary is being generated...")

        st.header("✨ Suggested Questions")
        if st.session_state.suggested_questions:
            for i, question in enumerate(st.session_state.suggested_questions):
                if st.button(f"{question}", key=f"q_{i}"):
                    with tab2:
                        response = handle_user_question(question)
        else:
            st.info("Suggested questions are being generated...")
        st.markdown("---")
    with tab2:
        st.header("💬 Interact with Your PDF (All questions answered with sources!)")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.chat_message("user", avatar="🧑‍💻").write(question)
            st.chat_message("assistant", avatar="🤖").write(answer)
        user_question = st.chat_input("💡 Ask anything about your PDF:")
        if user_question:
            st.chat_message("user", avatar="🧑‍💻").write(user_question)
            handle_user_question(user_question)
else:
    st.info("Please upload a PDF document to get started.")
    st.markdown("""
    **This application allows you to:**
    - Upload large PDFs (all types: scanned, photos, normal, >230 pages)
    - Get an automatic summary of the document
    - See suggested questions based on the content
    - Ask questions about the document using a chat interface
    - Download the chat history for future reference

    **Tip:** If your PDF is a scan/photo, OCR is automatic but may take longer.
    """)

st.markdown("""
---
<center>
    <sub>Made with 🤖 + 🔥 by Abhinav Viswanathula. Built on Streamlit, LangChain & Llama 3.<br>
    <b>Tip:</b> Lightning-fast for normal and scanned PDFs – every answer is referenced!
    </sub>
</center>
""", unsafe_allow_html=True)
