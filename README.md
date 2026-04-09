# ⚕ MediMind — Medical RAG Chatbot

> An AI-powered medical chatbot built with Retrieval-Augmented Generation (RAG), grounded in the GALE Encyclopedia of Medicine. Refuses out-of-scope questions — zero hallucination by design.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medimind-bot.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.26-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![LLaMA](https://img.shields.io/badge/LLM-LLaMA%203.1%208B-purple)

---

## 🔗 Live Demo

**[medimind-bot.streamlit.app](https://medimind-bot.streamlit.app/)**

---

## 🧠 What is MediMind?

MediMind is a domain-specific medical Q&A chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer questions strictly from a verified medical knowledge base. Unlike general-purpose LLMs, MediMind:

- ✅ Answers only from the **GALE Encyclopedia of Medicine**
- ✅ **Refuses out-of-scope questions** — no hallucination
- ✅ Returns **context-grounded responses** with LLaMA 3.1 via Groq
- ✅ Deployed and accessible publicly via Streamlit Cloud

---

## 🏗️ Architecture

```
User Query
    │
    ▼
HuggingFace Embeddings          ← all-MiniLM-L6-v2
    │
    ▼
FAISS Vector Store              ← semantic similarity search (top-k=3)
    │
    ▼
LangChain Retrieval Chain       ← context + query → prompt
    │
    ▼
LLaMA 3.1 8B (via Groq API)    ← fast inference
    │
    ▼
Grounded Response
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit (custom dark medical UI) |
| LLM | LLaMA 3.1 8B Instant via Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| RAG Framework | LangChain |
| Knowledge Base | GALE Encyclopedia of Medicine |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Harsh28-raj/MediMind-.git
cd MediMind-
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

**4. Run the app**
```bash
streamlit run medibot.py
```

---

## 📁 Project Structure

```
MediMind-/
├── medibot.py                  ← main Streamlit app
├── requirements.txt
├── .gitignore
├── vectorstore/
│   └── db_faiss/
│       ├── index.faiss         ← FAISS vector index
│       └── index.pkl
└── data/
    └── The_GALE_ENCYCLOPEDIA.pdf
```

---

## ✨ Key Features

- **Hallucination-free** — strictly answers from retrieved medical context
- **Semantic search** — finds relevant passages using vector similarity
- **Fast inference** — LLaMA 3.1 via Groq delivers sub-second responses
- **Dark medical UI** — custom Streamlit CSS with teal accent theme
- **Context-aware refusal** — politely declines non-medical questions

---

## 🔒 Hallucination Prevention

MediMind is designed with **strict RAG grounding**. If a question falls outside the medical knowledge base, the bot explicitly states it cannot answer — rather than generating a plausible but incorrect response.

```
User: "Who is the Prime Minister of India?"
Bot:  "Your question cannot be answered based on the provided context."
```

This is intentional and production-grade behavior.

---

## 👨‍💻 Author

**Harsh Raj**
- GitHub: [@Harsh28-raj](https://github.com/Harsh28-raj)
- 

---

## ⭐ Star this repo if you found it useful!
