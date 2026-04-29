# AI Digital Asset Manager (AI-DAM)

An intelligent AI-powered document management and conversational QA platform with **strict document-isolated RAG**, semantic search, FAISS vector indexing, and multimodal image recognition (CLIP + Tesseract OCR).

---

## 🚀 Quick Start for New Users / Developers

### 1. Prerequisites
- **Python**: 3.10 – 3.14 recommended
- **Git**: Installed on your system
- **Tesseract OCR** (Optional, only needed for image OCR text extraction):
  - **macOS**: `brew install tesseract`
  - **Ubuntu / Debian**: `sudo apt install tesseract-ocr`
  - **Windows**: Download installer from GitHub UB-Mannheim/tesseract

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/sumityadav026/Ai-Digital-Asset-Manager.git
cd AI_DAM_project

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r Requirements.txt
```

### 3. Configure API Keys

Get a free API key from [Groq Console](https://console.groq.com/keys) (powers fast document QA responses).

Export your key in your terminal or save it to your shell profile:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```
*(On Windows Command Prompt: `set GROQ_API_KEY=your_key`, PowerShell: `$env:GROQ_API_KEY="your_key"`)*

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 👥 How to Share with Others

### Option A: Share over the Same Local Wi-Fi / LAN
If someone is on the same local network (office/home Wi-Fi):
1. Start Streamlit bound to all interfaces:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
2. Find your local IP (e.g., `192.168.1.50` on macOS via `ipconfig getifaddr en0`).
3. Others can access the app directly from their browser at:
   `http://<YOUR_LOCAL_IP>:8501`

### Option B: Free Cloud Hosting (Streamlit Community Cloud)
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Select your repository, branch (`main`), and main file (`app.py`).
4. In Advanced Settings, add your secret:
   ```
   GROQ_API_KEY = "your_actual_key"
   ```
5. Click **Deploy** — Streamlit generates a public shareable URL (e.g. `https://your-dam.streamlit.app`).

### Option C: Instant Temporary Public Link (Localtunnel / Ngrok)
To let someone anywhere access your running app temporarily:
```bash
npx localtunnel --port 8501
```
Send the generated `https://xxxx.loca.lt` URL to the other person.

---

## 📖 Application Features & User Guide

| Section | What It Does & How to Use |
|---|---|
| **📊 Dashboard** | Provides quick metrics (number of indexed documents, chunks, vectors) and quick navigation cards to search or chat with recent documents. |
| **📄 My Documents** | **Upload**: Drop any `.pdf` or `.docx` file. Automatic chunking, embedding generation, and MD5 deduplication.<br>**Library**: Filter, sort by size/chunks, view extracted text, or launch search/chat. |
| **💬 Document Chat** | Dedicated conversation interface with **strict document isolation**. Every answer is strictly grounded in the selected document (no cross-document contamination). Supports quick suggestion chips, conversation history, and source verification citations. |
| **🔍 Search** | Global semantic hybrid search across all indexed files. Ranks results by semantic similarity and keyword coverage. |
| **🖼 Image Recognition** | Upload `.png`, `.jpg`, or `.jpeg` images. Uses OpenAI CLIP for zero-shot object classification and Tesseract OCR for text extraction. |
| **⚙ Settings** | Displays system status, embedding model configuration, and database rebuild instructions. |

---

## 🧪 Running Automated Tests

To run the complete test suite including cross-document retrieval isolation regression tests:

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## 🛠 Rebuilding the Index
If you ever upload files directly into the `uploads/` folder or change embedding parameters, rebuild the database with:
```bash
python rebuild_index.py
```
