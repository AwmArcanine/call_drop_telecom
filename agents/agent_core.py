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

import re
from typing import Dict, List

def fallback_recs(region: str, metrics: Dict, meta: Dict = None) -> List[str]:
    """
    deterministic, engineering-focused fallback recommendations.
    Not a placeholder — each item is action-oriented and includes steps/timeframe.
    """
    recs = []
    tower = (meta or {}).get("Tower_ID") or "target tower"
    sig = metrics.get("avg_signal")
    congestion = str(metrics.get("congestion_level") or "").lower()
    handoff = float(metrics.get("handoff_pct") or 0)
    drops = float(metrics.get("drop_rate") or 0)

    # signal-focused
    if sig is not None and float(sig) < -90:
        recs.append(
            f"Optimize RF coverage at {tower}: perform antenna re-tilt (+2° → +4°) on affected sectors, "
            "increase TX power by up to 1–2 dB if allowed, then run a targeted drive test within 7 days "
            "and collect RSRP/RSRQ/SINR to validate impact. Priority: High."
        )

    # congestion-focused
    if "high" in congestion or drops > 5:
        recs.append(
            "Reduce congestion through capacity and scheduler actions: deploy 1 microcell in the top hotspot "
            "or enable carrier aggregation on busiest carriers; tune scheduler (increase PRB allocation for voice) "
            "and re-run load tests during peak hours. Start with 1 cell and evaluate after 2 weeks. Priority: High."
        )

    # handoff-focused
    if handoff > 8:
        recs.append(
            "Optimize handoff performance: review neighbor lists, reduce hysteresis by 0.5–2 dB and time-to-trigger "
            "to 40–80 ms where appropriate, then monitor handover success metrics for 72 hours. Priority: Medium."
        )

    # generic stability / diagnostics
    if len(recs) < 3:
        recs.append(
            "Run a targeted root-cause validation: schedule drive tests, sample logs (RSRP/RSRQ/SINR, load, KPIs), "
            "and compare worst-performing sectors to neighboring sectors. Use results to select targeted fixes. Priority: Medium."
        )

    # ensure uniqueness and reasonable length
    unique = []
    for r in recs:
        if r not in unique:
            unique.append(r)
    return unique[:5]



# --- AI Recommendations ---
def generate_ai_recommendations(region: str, metrics: Dict) -> str:
    """
    Generates realistic, region-specific telecom optimization actions based on root causes.
    - Analyses signal, congestion, and handoff issues dynamically.
    - Produces diverse, non-repetitive, actionable recommendations.
    """
    signal = metrics.get('avg_signal')
    congestion = str(metrics.get('congestion_level')).lower()
    handoff = float(metrics.get('handoff_pct') or 0)
    drop = float(metrics.get('drop_rate') or 0)

    # --- Step 1: derive root issues ---
    issues = []
    if signal and float(signal) < -90:
        issues.append("weak signal coverage")
    if "high" in congestion:
        issues.append("network congestion")
    if handoff > 10:
        issues.append("high handoff failure rate")
    if not issues:
        issues.append("minor optimization required")

    context = ", ".join(issues)

    # --- Step 2: construct intelligent prompt ---
    prompt = (
        f"You are a telecom optimization engineer analyzing network issues in {region}.\n"
        f"Detected issues: {context} (signal={signal} dBm, congestion={congestion}, handoff failure={handoff}%).\n"
        f"Dropout rate observed: {drop}%.\n\n"
        "Provide 2–4 **practical and specific** technical actions engineers can take to resolve these problems.\n"
        "Each recommendation must:\n"
        "- Be unique and directly related to one or more detected issues.\n"
        "- Include a Priority (High/Medium/Low).\n"
        "- Be concise (max 2 lines).\n\n"
        "Format strictly as:\n"
        "1. <Action> - Priority: <High/Medium/Low> - <Short reason>\n"
        "2. ...\n"
        "3. ...\n"
        "Generate only the numbered recommendations, no explanations."
    )

    # --- Step 3: generate ---
    response = llm(
        prompt,
        max_new_tokens=300,
        temperature=0.55,
        top_p=0.9,
        repetition_penalty=1.25,
        do_sample=True,
    )[0]["generated_text"].strip()

    # --- Step 4: clean formatting ---
    import re
    recs = re.findall(r"\d\..*?(?=\d\.|$)", response, re.DOTALL)
    recs = [r.strip() for r in recs if len(r.strip()) > 10]
    seen = set()
    recs = [r for r in recs if not (r in seen or seen.add(r))]

    # --- Step 5: AI fallback (minimal) ---
    if not recs:
        print("⚙️ AI fallback triggered — generating smart defaults.")
        recs = []
        if signal and float(signal) < -90:
            recs.append("1. Optimize antenna tilt and transmission gain - Priority: High - To improve weak signal zones.")
        if "high" in congestion:
            recs.append("2. Deploy microcells or redistribute load - Priority: High - To ease traffic congestion.")
        if handoff > 10:
            recs.append("3. Adjust handoff thresholds - Priority: Medium - To minimize frequent handoffs.")
        if not recs:
            recs.append("1. Conduct routine network optimization drive tests - Priority: Medium - To maintain stable connectivity.")

    return "\n".join(recs[:4])


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
