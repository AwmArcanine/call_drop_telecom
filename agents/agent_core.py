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
def generate_ai_recommendations(region: str, metrics: Dict, root_cause: str = None) -> str:
    """
    Generates 3 clear, context-aware recommendations derived from the root cause.
    Avoids numeric or hardcoded values. Each suggestion is distinct and practical.
    """
    signal = metrics.get("avg_signal")
    congestion = metrics.get("congestion_level")
    handoff = metrics.get("handoff_pct")
    drop_rate = metrics.get("drop_rate")

    context_summary = (
        f"Average signal: {signal} dBm, Congestion: {congestion}, "
        f"Handoff failures: {handoff}%, Drop rate: {drop_rate}%"
    )

    rc_text = root_cause or "Weak signal and congestion detected in the region."

    # Strongly structured prompt
    prompt = (
        f"You are a telecom optimization specialist analyzing network data for {region}.\n"
        f"Root Cause Summary: {rc_text}\n"
        f"Context Metrics: {context_summary}\n\n"
        "Based on this information, provide three specific, high-level engineering actions "
        "that can reduce call drops and improve network reliability.\n\n"
        "Guidelines:\n"
        "- Avoid numeric values or tower IDs.\n"
        "- Make each recommendation unique and address a different issue (signal, congestion, or handoff).\n"
        "- Each must follow this exact format:\n"
        "1. <Action> - Priority: <High/Medium/Low> - <Short rationale>\n"
        "2. <Action> - Priority: <High/Medium/Low> - <Short rationale>\n"
        "3. <Action> - Priority: <High/Medium/Low> - <Short rationale>\n"
        "Ensure the reasoning clearly connects to the observed problem."
    )

    try:
        response = llm(
            prompt,
            max_new_tokens=250,
            temperature=0.45,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
        )[0]["generated_text"].strip()
    except Exception as e:
        print(f"⚠️ Model error: {e}")
        response = ""

    import re
    recs = re.findall(r"^\s*\d+\.\s*.*?(?=\d+\.\s*|$)", response, flags=re.MULTILINE | re.DOTALL)
    recs = [r.strip() for r in recs if len(r.strip()) > 15 and "action" not in r.lower()]

    # Intelligent fallback (if model output is too vague or repetitive)
    if len(recs) < 3 or len(set(recs)) < 3:
        recs = []
        if signal and float(signal) < -85:
            recs.append(
                "1. Enhance coverage through antenna orientation and power optimization - Priority: High - To improve weak signal strength in low-RSRP zones."
            )
        if "high" in str(congestion).lower():
            recs.append(
                "2. Balance user load across nearby cells or deploy small cells in dense areas - Priority: High - To mitigate congestion during peak hours."
            )
        if handoff and float(handoff) > 10:
            recs.append(
                "3. Fine-tune handoff thresholds and neighboring cell configurations - Priority: Medium - To reduce call drops caused by failed transitions."
            )
        if len(recs) < 3:
            recs.append(
                "3. Conduct field validation and network audits - Priority: Medium - To ensure coverage and connectivity meet operational standards."
            )

    # Final cleanup
    final_recs = []
    seen = set()
    for r in recs:
        clean = re.sub(r"^\d+\.\s*", "", r).strip()
        if clean and clean not in seen:
            seen.add(clean)
            final_recs.append(clean)
    numbered = [f"{i+1}. {r}" for i, r in enumerate(final_recs[:3])]
    return "\n".join(numbered)


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
