# Agent-Based Call Drop Analysis (Telecom)

Overview
--------
Automates root-cause analysis of call drops using retrieval-augmented generation (RAG) and an LLM agent. Built with open-source models and tools: Python, LangChain/LlamaIndex, SentenceTransformers, Chroma/FAISS, and Streamlit.

Goals
-----
- Retrieve relevant log snippets from vector DB.
- Identify probable causes of call drops for a region.
- Produce human-readable explanation + supporting evidence.
- Recommend concrete remediation steps (rules + LLM-assisted).

Stack
-----
- Python 3.10+
- LangChain or LlamaIndex (examples use LangChain-style interfaces)
- SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings
- ChromaDB (or FAISS) as vector store
- HuggingFace transformers (local/remote open-source chat models)
- Streamlit for UI

Quick start
-----------
1. Create a virtualenv and install requirements:
2. Place `telecom_logs.csv` in `data/`.
3. Build vector DB:
4. Run Streamlit app:

Deliverables
------------
- `scripts/build_vector_db.py` — cleans, chunks, and embeds logs into Chroma.
- `agents/agent_core.py` — defines LangChain agent + tools.
- `app/streamlit_app.py` — UI to query agent and get report.
- Sample dataset: `data/telecom_logs.csv`

Acceptance criteria
-------------------
- Query a region (e.g., "Why are call drops high in Hyderabad?") returns:
- Short root cause summary
- Evidence lines with original log snippets
- 3-5 recommended remediation steps
- Recommendations include at least one rule-based suggestion (e.g., if congestion `High` -> suggest microcells)
- All pieces run locally with open-source models (no fine-tuning)
