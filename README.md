# 🧠 Startup Pitch Deck Chatbot

This is a simple AI agent that allows users to upload a startup pitch deck (PDF) and ask questions about it — powered by Mistral 7B, RAG, and ChromaDB vector search.

---

## 🚀 Features

- Upload a pitch deck PDF once
- Ask multiple natural language questions about it
- Uses RAG: text extraction, chunking, embedding, and retrieval
- Vector DB: [ChromaDB](https://www.trychroma.com/)
- Embedding: [SentenceTransformers MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- UI: Simple and clean HTML with Flask backend

---

## 📂 File Structure

startup-pitch-chatbot/

├── app.py

├── requirements.txt

├── templates/

│   └── index.html

├── rag_core/

│   ├── embed_store.py

│   └── retriever.py

├── uploads/           # (auto-created, stores PDFs)

├── .gitignore

└── README.md

---

## ⚙️ Setup Instructions

```bash



# Clone the repo
git clone https://github.com/your-username/startup-pitch-chatbot.git
cd startup-pitch-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the app
python app.py
---
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.



## 📦 Requirements

Install using: 
pip install flask chromadb sentence-transformers PyMuPDF
```


## 🧠 How it Works

1. **Upload PDF** → extracts and chunks text
2. **Embeds Chunks** → using `MiniLM-L6-v2`
3. **Stores in Vector DB** → using `ChromaDB`
4. **Ask Questions** → retrieves relevant chunks
5. **Shows Answer**
