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

PRIMARY_MODEL = "MBZUAI/LaMini-Flan-T5-783M"  # Fast + reasoning capable
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
def generate_ai_recommendations(region: str, metrics: Dict) -> str:
    """
    Generates targeted and data-driven recommendations directly from network metrics.
    The AI now reasons over signal, congestion, and handoff data instead of following a template.
    Produces 3–5 distinct actions that align with the root cause.
    """

    signal = metrics.get('avg_signal')
    congestion = str(metrics.get('congestion_level')).lower()
    handoff = float(metrics.get('handoff_pct') or 0)
    drop = float(metrics.get('drop_rate') or 0)

    # Build factual context for reasoning
    prompt = (
        f"You are a telecom optimization expert analyzing network performance in {region}.\n"
        "Here are the detected metrics:\n"
        f"- Average Signal Strength: {signal} dBm\n"
        f"- Congestion Level: {congestion}\n"
        f"- Handoff Failure Rate: {handoff}%\n"
        f"- Dropout Rate: {drop}%\n\n"
        "Analyze these metrics carefully and infer the **root causes** of call drops. "
        "Then provide 3–5 **unique and technically detailed recommendations** that directly address those causes.\n"
        "Each recommendation must include:\n"
        "- A clear technical action (not a generic statement)\n"
        "- A realistic priority (High/Medium/Low)\n"
        "- A short reason grounded in the metrics above\n\n"
        "Be concise but data-driven.\n"
        "Output format:\n"
        "1. <Action> - Priority: <High/Medium/Low> - <Reason>\n"
        "2. <Action> - Priority: <High/Medium/Low> - <Reason>\n"
        "3. ... (up to 5 suggestions if relevant)"
    )

    response = llm(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.5,        # allow creativity for diversity
        top_p=0.9,
        repetition_penalty=1.2,
    )[0]["generated_text"].strip()

    import re
    recs = re.findall(r"\d\.\s?.*?(?=\d\.|$)", response, re.DOTALL)
    recs = [r.strip() for r in recs if len(r.strip()) > 10]

    # --- Intelligent fallback ---
    if not recs:
        print("⚙️ Fallback: Generating logic-based suggestions.")
        recs = []
        i = 1

        if signal and float(signal) < -85:
            recs.append(f"{i}. Reorient antennas and increase transmission gain - Priority: High - Weak signal ({signal} dBm) affects coverage area.")
            i += 1
        if "high" in congestion:
            recs.append(f"{i}. Add small cells or optimize scheduler settings - Priority: High - Congestion detected in {region}.")
            i += 1
        if handoff > 10:
            recs.append(f"{i}. Fine-tune handoff thresholds and reconfiguration timers - Priority: Medium - {handoff}% handoff failure rate detected.")
            i += 1
        if drop > 5:
            recs.append(f"{i}. Optimize frequency reuse and interference management - Priority: Medium - High dropout rate ({drop}%) observed.")
            i += 1
        if len(recs) < 3:
            recs.append(f"{i}. Conduct drive tests and parameter audits - Priority: Medium - To ensure optimized performance under varying load.")

    # Cap between 3–5
    return "\n".join(recs[:5])





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
