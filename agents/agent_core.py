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
    Produce 3 concise, non-hardcoded recommendations tied to the root cause.
    The model should produce high-level engineering actions (no fixed values).
    """
    # Prepare metric/context summary (human-readable)
    signal = metrics.get("avg_signal")
    congestion = metrics.get("congestion_level")
    handoff = metrics.get("handoff_pct")
    drop_rate = metrics.get("drop_rate")

    context_lines = []
    if signal is not None:
        context_lines.append(f"Average signal: {signal} dBm")
    if congestion:
        context_lines.append(f"Congestion level: {congestion}")
    if handoff is not None:
        context_lines.append(f"Handoff failures: {handoff}%")
    if drop_rate is not None:
        context_lines.append(f"Dropout rate: {drop_rate}%")
    context_summary = "; ".join(context_lines) if context_lines else "No numeric metrics available."

    # Root cause short summary (prefer explicit root_cause text if passed)
    rc_text = (root_cause.strip() if root_cause else "No explicit root cause provided.")
    
    # Prompt emphasizes *no hardcoded numbers*, actionable but general steps
    prompt = (
        f"You are a telecom optimization engineer advisor. Analyze the following situation and suggest 3 "
        f"distinct, practical, non-numeric, engineering actions that field or RAN engineers can take.\n\n"
        f"Region: {region}\n"
        f"Root cause summary: {rc_text}\n"
        f"Context metrics: {context_summary}\n\n"
        "Requirements for the output:\n"
        "- Return exactly three numbered recommendations (1., 2., 3.).\n"
        "- Each recommendation should be an actionable, non-hardcoded instruction (no fixed parameter values, no site IDs).\n"
        "- Each recommendation must include a one-line rationale tied to the root cause/context.\n"
        "- Use engineering language but keep it concise (1-2 lines per recommendation).\n\n"
        "Example (bad): 'Retilt antenna by +3° on tower T951.'   <-- DO NOT do this.\n"
        "Example (good): 'Adjust antenna orientation and power to improve coverage in the weak-signal zones. - Rationale: weak RSRP and high call drops near cell edge.'\n\n"
        "Now generate the 3 recommendations."
    )

    # Ask the LLM
    try:
        resp = llm(
            prompt,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.45,
            top_p=0.9,
            repetition_penalty=1.05,
        )[0]["generated_text"].strip()
    except Exception as e:
        # If the model call fails, fall back to logic-based text suggestions
        print(f"⚠️ LLM error in generate_ai_recommendations: {e}")
        resp = ""

    # Simple extraction of lines starting with numbers
    import re
    recs = re.findall(r"^\s*\d+\.\s*(.+)$", resp, flags=re.MULTILINE)
    recs = [r.strip() for r in recs if len(r.strip()) > 10]

    # If LLM didn't return the requested numbered lines, try splitting heuristically
    if len(recs) < 3:
        # try to pull any lines separated by blank lines or sentence boundaries
        candidates = [s.strip() for s in re.split(r"\n{1,}|\.\s+", resp) if len(s.strip()) > 10]
        # keep up to 3 unique, non-placeholder suggestions
        for c in candidates:
            if len(recs) >= 3:
                break
            if "action" in c.lower() or "recommend" in c.lower() or any(word in c.lower() for word in ["antenna", "microcell", "handoff", "drive test", "backhaul", "balanc"]):
                recs.append(c)
    
    # Final logic-guided fallback if still insufficient: generate non-hardcoded steps based on detected issues
    if len(recs) < 3:
        print("⚙️ Recommendation fallback: using rule-guided phrasing.")
        fallback = []
        # prioritize by detected root cause keywords
        rc_lower = rc_text.lower()
        if "signal" in rc_lower or (signal is not None and float(signal) < -90):
            fallback.append("Optimize RF coverage (adjust antenna orientation, downtilt/uptilt strategy, or power settings) and validate via targeted drive tests - Rationale: to improve user RSRP and reduce edge-area call drops.")
        if "congest" in rc_lower or (isinstance(congestion, str) and "high" in congestion.lower()) or (drop_rate and drop_rate > 2):
            fallback.append("Implement capacity and load management (e.g., enable load balancing, offload heavy traffic to small cells or carriers) and monitor PRB/throughput during peaks - Rationale: to reduce congestion-driven call drops.")
        if "handoff" in rc_lower or (handoff and float(handoff) > 5):
            fallback.append("Tune handoff/handover strategy and neighbor relations (review timers, margins, and neighbor lists) and validate mobility scenarios - Rationale: to reduce failures during cell transitions.")
        # generic if still less than 3
        if len(fallback) < 3:
            fallback.append("Schedule routine drive tests and targeted KPI monitoring for the affected hours to validate the impact of any changes - Rationale: ensures fixes are verified in real conditions.")
        # collect up to 3
        recs = fallback[:3]

    # Final normalization: ensure no placeholder tokens and concise phrasing
    normalized = []
    for i, r in enumerate(recs[:3], start=1):
        # remove any residual numbering from LLM and enforce format "1. <text> - Rationale: ..."
        text = re.sub(r"^\s*\d+\.\s*", "", r).strip()
        # if the text already contains a rationale separated by '-' or '—', keep it, else append generic label
        if "-" in text:
            final = f"{i}. {text}"
        else:
            final = f"{i}. {text} - Rationale: related to the detected root cause."
        normalized.append(final)

    return "\n".join(normalized)


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
