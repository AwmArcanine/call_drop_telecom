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
    Generates precise, region-aware technical recommendations for telecom engineers.
    The output uses realistic optimization actions (antenna tilt, small cells, TTT tuning, etc.)
    based on the root cause and metrics observed.
    """
    signal = metrics.get("avg_signal")
    congestion = str(metrics.get("congestion_level")).lower()
    handoff = float(metrics.get("handoff_pct") or 0)
    drop = float(metrics.get("drop_rate") or 0)

    issues = []
    if signal and float(signal) < -90:
        issues.append("weak signal strength")
    if "high" in congestion:
        issues.append("high congestion")
    if handoff > 10:
        issues.append("frequent handoff failures")
    if not issues:
        issues.append("minor network performance variations")

    issue_summary = ", ".join(issues)

    # More structured, technically grounded prompt
    prompt = (
        f"You are a telecom network optimization engineer analyzing call drop logs for region {region}.\n"
        f"Detected conditions:\n"
        f"- Average Signal Strength: {signal} dBm\n"
        f"- Congestion Level: {congestion}\n"
        f"- Handoff Failure Rate: {handoff}%\n"
        f"- Dropout Rate: {drop}%\n\n"
        f"Root cause summary: {issue_summary}.\n\n"
        f"Generate exactly 3–5 region-specific recommendations with measurable actions engineers can implement.\n"
        f"Each recommendation must mention specific network parameters, equipment, or configuration changes.\n"
        f"Use the following format:\n"
        f"1. <Action> - Priority: <High/Medium/Low> - <Technical justification>\n\n"
        f"Example recommendations:\n"
        f"1. Increase antenna tilt by +2° for tower T951 in {region} to improve coverage in weak-signal zones. - Priority: High - Signal observed at -95 dBm.\n"
        f"2. Deploy an additional microcell in {region} near the congested cluster to reduce PRB load by 20%. - Priority: High - Congestion detected during evening hours.\n"
        f"3. Adjust handoff Time-To-Trigger (TTT) from 80 ms to 120 ms for smoother cell transitions. - Priority: Medium - 13% handoff failures between adjacent cells.\n\n"
        f"Now generate new, unique, technically valid recommendations for this region only:"
    )

    response = llm(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
    )[0]["generated_text"].strip()

    import re
    recs = re.findall(r"\d\.\s?.*?(?=\d\.|$)", response, re.DOTALL)
    recs = [r.strip() for r in recs if len(r.strip()) > 10]

    # If AI output is invalid or generic, create a logic-based fallback
    if len(recs) < 3:
        print("⚙️ Fallback: Using logic-based engineering recommendations.")
        recs = []
        i = 1
        if signal and float(signal) < -85:
            recs.append(f"{i}. Retilt antennas for low-coverage sectors in {region} by +3° and verify RSRP improvement. - Priority: High - Weak signal around {signal} dBm.")
            i += 1
        if "high" in congestion:
            recs.append(f"{i}. Add one microcell near dense zones in {region} or enable load balancing on nearby towers. - Priority: High - Heavy congestion detected.")
            i += 1
        if handoff > 10:
            recs.append(f"{i}. Increase handoff margin and TTT by 30–50 ms to reduce drop rate caused by premature transitions. - Priority: Medium - {handoff}% handoff failure rate.")
            i += 1
        if drop > 5:
            recs.append(f"{i}. Upgrade backhaul link and monitor throughput on high-traffic nodes. - Priority: Medium - Drop rate {drop}%.")
            i += 1
        if len(recs) < 3:
            recs.append(f"{i}. Conduct a field drive test in {region} to verify coverage and signal overlap. - Priority: Medium - To validate configuration changes.")
    
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
