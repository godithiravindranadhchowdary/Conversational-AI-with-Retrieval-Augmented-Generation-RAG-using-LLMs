# RAG AI PDF BEAST — RAG-powered PDF Chatbot

A Streamlit app that uses Retrieval-Augmented Generation (RAG) to summarize and chat with PDF documents. Upload a PDF (including scanned/image PDFs), get an automatic summary, suggested starter questions, and ask follow-up questions answered with sourced excerpts.

Key implementation details are in [app.py](app.py).

## Features

- Upload and process large PDFs (supports scanned documents via OCR).
- Automatic document summarization (5–10 key points).
- Auto-generated suggested questions (5–7 starter questions).
- Interactive chat that answers using retrieved document context and returns source excerpts.
- Uses FAISS for vector search and `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Uses Groq's `ChatGroq` (Llama 3) as the LLM for summarization and Q&A.

## Requirements

- Python 3.8+
- `requirements.txt` (install with `pip install -r requirements.txt`)
- Internet access for the Groq API
- Poppler (for `pdf2image`) and Tesseract OCR installed on the host machine
- A Groq API key (set in environment or `.env`)

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1   # PowerShell (Windows)
# or: .venv\\Scripts\\activate    # cmd.exe (Windows)
# or: source .venv/bin/activate   # macOS / Linux
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Install native OCR dependencies:

- Poppler (pdf2image): Windows users can download prebuilt binaries (e.g. from https://github.com/oschwartz10612/poppler-windows) and set `POPPLER_PATH` in `app.py` or add Poppler's `bin` folder to your `PATH`.
- Tesseract OCR: Install from https://github.com/tesseract-ocr/tesseract and ensure the executable is reachable. The app sets `pytesseract.pytesseract.tesseract_cmd` to the `TESSERACT_PATH` constant in `app.py` by default.

You can edit `POPPLER_PATH` and `TESSERACT_PATH` directly in `app.py` or modify your system `PATH`.

## Configuration

- Create a `.env` file next to `app.py` (project root) or export environment variables. Example:

```
GROQ_API_KEY=your_groq_api_key_here
```

Note: `app.py` uses `python-dotenv` to load `.env`. The code currently contains a fallback/default `GROQ_API_KEY` value — replace it with your own key for production use and remove any hardcoded secrets.

## Run the App

From the folder that contains `app.py` run:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## How it works (brief)

- PDF text is loaded via `PyPDFLoader`.
- If no selectable text is found, the app runs OCR: `pdf2image` converts pages to images (requires Poppler) and `pytesseract` extracts text.
- The text is split into overlapping chunks via `RecursiveCharacterTextSplitter`.
- Embeddings are created with `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`), stored in FAISS, and used to retrieve relevant context for questions.
- `ChatGroq` (Groq API / Llama 3) is used for summarization, generating suggested questions, and answering user queries in a conversational retrieval chain.

## Usage

1. Upload a PDF using the sidebar uploader.
2. Wait until the app processes the file (OCR will run automatically on scanned PDFs).
3. Read the generated summary and suggested questions.
4. Ask questions using the chat input — answers include page-sourced excerpts when available.
5. Download chat history from the sidebar if desired.

## Troubleshooting

- OCR errors ("Poppler not found" or `pdf2image` failures): Ensure Poppler is installed and `POPPLER_PATH` is set correctly or the bin folder is in your `PATH`.
- Tesseract errors (`pytesseract` can't find tesseract): Install Tesseract and update `TESSERACT_PATH` in `app.py` or add it to `PATH`.
- Groq API authentication errors: Verify `GROQ_API_KEY` in `.env` or environment variables; check your Groq account and quotas.
- Slow processing / high memory usage: For very large PDFs, processing and embedding can be resource intensive. Try smaller PDFs, increase system RAM, or adjust `chunk_size` in `app.py`.
- Empty summary or no chunks: Some PDFs are encrypted or have unusual structure—try opening them with a PDF reader or run OCR manually to confirm content.

## Security & Privacy

- Keep API keys out of source control. Use `.env` and a secrets manager for production.
- Do not index or upload sensitive personal data without appropriate consent and protections.

## File references

- Main app: [app.py](app.py)
- Requirements: [requirements.txt](requirements.txt)

## Credits

Built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and Groq's ChatGroq LLM. Original styling and UI by Abhinav Viswanathula.

---

If you want, I can also:
- Add a short `CONTRIBUTING.md` with development steps,
- Add a `.env.example` file,
- Or adjust the README to show exact commands for your OS (Windows / macOS / Linux).

