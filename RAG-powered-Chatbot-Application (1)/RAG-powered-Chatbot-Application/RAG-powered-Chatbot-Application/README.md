# RAG-powered PDF Chatbot Application

A Streamlit-based Generative AI Chatbot that leverages Retrieval-Augmented Generation (RAG) to process, summarize, and answer questions about large PDF documents (230+ pages), integrating LLMs and vector search for an interactive exploration experience.

## Features

- **Support for Large PDFs**: Handles massive documents (230+ pages) with efficient memory usage and chunking.
- **Automatic Document Summarization**: Provides a summary with 5–10 key points after PDF upload.
- **Smart Suggested Questions**: Generates 5–10 insightful questions based on the document.
- **Interactive Chat Interface**: Chat directly with your document using natural language.
- **Vector Database Integration**: Employs FAISS for efficient storing and searching of embeddings.
- **Advanced AI**: Utilizes Hugging Face sentence transformers for embedding, and Groq’s Llama 3 (70B) as the language model.
- **User-Friendly**: Clean Streamlit interface with tabs for document summary, suggested questions, and chat.
- **Downloadable Chat History**: Export your Q&A sessions for future reference.

## Setup Instructions

### 1. Clone the Repository

```
git clone <repository-url>
cd rag-pdf-chatbot
```

### 2. Create a Virtual Environment

```
python -m venv venv
# On Unix/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can obtain a Groq API key by signing up at the official Groq Console.

### 5. Run the Application

```
streamlit run app.py
```

## Usage Guide

1. **Upload** a PDF using the sidebar uploader.
2. **Wait** for processing (OCR will automatically run on scanned/image PDFs).
3. **Summary & Questions**: View the “Document Summary” and “Suggested Questions” tabs.
4. **Chat**: Switch to the “Chat” tab to ask your own questions about the content.
5. **Download**: Export your conversation history from the sidebar.

## Architecture Overview

| Component         | Description                                                                          |
|-------------------|--------------------------------------------------------------------------------------|
| PDF Processing    | PyPDFLoader extracts PDF text; OCR via Tesseract is applied for scanned/image files. |
| Text Splitting    | RecursiveCharacterTextSplitter breaks text into manageable overlapping chunks.       |
| Embeddings        | Uses Hugging Face model (`all-MiniLM-L6-v2`) for creating document embeddings.       |
| Vector Storage    | FAISS stores and enables fast similarity searching of embeddings.                    |
| Language Model    | Groq’s Llama 3 (70B) provides intelligent, contextual responses.                     |
| User Interface    | Streamlit offers a clean, responsive web interface.                                  |

## Requirements

- Python 3.8 or newer
- Internet connection (to use the Groq API)
- Groq API key (register at Groq Console)
- At least 8 GB RAM recommended
- Poppler (for pdf2image OCR) and Tesseract OCR installed with system PATH configured

## Limitations

- Processing extremely large PDFs (>500 pages) may take additional time and memory.
- API usage may be subject to rate limits per your Groq plan.
- Summaries are based on the initial sections of the document, not a full-document overview.

## Troubleshooting

| Problem                                | Solution                                                      |
|-----------------------------------------|---------------------------------------------------------------|
| Memory issues with large files          | Try a smaller PDF or lower the text chunk size in code.        |
| API key not working                     | Ensure your `.env` file is correct and restart the application.|
| Slow response or OCR errors             | Confirm Poppler & Tesseract installation and system PATH setup.|
| API calls failing                       | Check your internet connection and Groq API quota.             |

**Tip:**  
Scanned PDFs are supported via automatic OCR, but processing time will increase depending on PDF length and scan quality.

For questions, contributions, or support, please open an issue or discussion in the repository.
```

Copy the entire content and paste it into your `README.md` file. Let me know if you'd like this as downloadable file output or GitHub-flavored enhancements.
