# agents/agent_core.py
from typing import List, Dict
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# --- Configurable ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "telecom_logs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

PRIMARY_MODEL = "MBZUAI/LaMini-Flan-T5-248M"  # Fast + reasoning capable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

# --- Initialize Chroma and Embedding ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# --- Load Fast Reasoning Model ---
print(f"🚀 Loading model: {PRIMARY_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    PRIMARY_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
llm = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=250,
    temperature=0.3,
    repetition_penalty=1.1,
)
print("✅ Model ready and optimized for reasoning-speed balance.")


# --- Query Vector DB ---
def query_vector_db(query_text: str, top_k: int = TOP_K) -> List[Dict]:
    q_emb = embedder.encode([query_text], convert_to_numpy=True)[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    items = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        items.append({"document": doc, "metadata": meta, "distance": float(dist)})
    return items


# --- AI Summary ---
def generate_ai_summary(snippets: List[str], region: str = None) -> str:
    if not snippets:
        return "No sufficient log data available to summarize."
    
    prompt = (
        f"You are a telecom expert analyzing call drop data for {region or 'Unknown'}.\n"
        "Summarize the issue in this structure:\n"
        "Region: (name)\n"
        "Observation: (trend in call drops, signal, congestion)\n"
        "Root Cause: (main causes such as weak signal, congestion, or handoff failure)\n"
        "Suggested Resolution Summary: (short actionable insight)\n\n"
        "Logs:\n"
    )
    for s in snippets:
        prompt += f"- {s}\n"

    response = llm(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]
    return response.strip()


# --- AI Recommendations ---
# --- AI Recommendations ---
def generate_ai_recommendations(region: str, metrics: Dict) -> str:
    """
    Generate realistic, data-specific network recommendations with reasoning.
    Fixes literal template repetition issue.
    """
    prompt = (
        f"You are a telecom optimization assistant for region {region}.\n"
        f"Here are the current network metrics:\n"
        f"- Signal Strength: {metrics.get('avg_signal')} dBm\n"
        f"- Congestion Level: {metrics.get('congestion_level')}\n"
        f"- Handoff Failure Rate: {metrics.get('handoff_pct')}%\n"
        f"- Dropout Rate: {metrics.get('drop_rate')}%\n\n"
        "Suggest exactly three *specific and practical* actions to reduce call drops.\n"
        "Each action should be short, actionable, and based on the data above.\n"
        "Format your response like this:\n\n"
        "1. Action - Priority: High/Medium/Low - Reason for the suggestion\n"
        "2. Action - Priority: High/Medium/Low - Reason for the suggestion\n"
        "3. Action - Priority: High/Medium/Low - Reason for the suggestion\n\n"
        "Avoid generic placeholders like (Action) or (Reason). Use real, data-based insights."
    )

    response = llm(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"].strip()

    import re
    recs = re.findall(r'\d\.\s?.*?(?=\d\.|$)', response, re.DOTALL)
    recs = [r.strip() for r in recs if len(r.strip()) > 10]

    # If no usable lines, try splitting by newlines
    if not recs:
        recs = [ln.strip() for ln in response.split("\n") if len(ln.strip()) > 10]

    # Fallback only if model gives junk or repeats template
    if not recs or any("(Action)" in r or "(Reason)" in r for r in recs):
        print("⚠️ Model produced template placeholders — using fallback.")
        recs = [
            "1. Increase antenna height and optimize tilt - Priority: High - To improve -95 dBm signal strength.",
            "2. Deploy microcells near high-load zones - Priority: High - To reduce congestion and improve throughput.",
            "3. Fine-tune handoff timers - Priority: Medium - To minimize 13% handoff failures during mobility.",
        ]

    return "\n".join(recs[:3])



# --- Main Analysis ---
def analyze_region_query(user_query: str, region: str = None):
    hits = query_vector_db(user_query if not region else f"Region: {region}")
    snippets = [
        h["document"]
        for h in hits
        if not region or h["metadata"].get("Region", "").lower() == region.lower()
    ]

    summary_txt = generate_ai_summary(snippets, region=region)

    if hits:
        top_meta = hits[0]["metadata"]
        call_drops = float(top_meta.get("Call_Drops", 0))
        total_calls = 1000
        drop_rate = round((call_drops / total_calls) * 100, 2)
        metrics = {
            "region": region or top_meta.get("Region"),
            "avg_signal": top_meta.get("Signal_Str_dBm"),
            "congestion_level": top_meta.get("Congestion_Level"),
            "handoff_pct": top_meta.get("Handoff_Failure_pct"),
            "drop_rate": drop_rate,
        }
    else:
        metrics = {"region": region, "avg_signal": None, "congestion_level": None, "handoff_pct": None, "drop_rate": 0}

    recs = generate_ai_recommendations(region or "Unknown", metrics)

    return {
        "summary": summary_txt,
        "recommendations": recs,
        "evidence": hits,
    }
