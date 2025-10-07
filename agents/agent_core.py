# agents/agent_core.py
from typing import List, Dict
import os
import json
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# --- Configurable ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "telecom_logs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

HF_MODEL_NAME = "google/flan-t5-large"

hf_token = os.getenv("HUGGINGFACE_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5


# --- Initialize Chroma and Embedding ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# --- Try to load Mistral (preferred), else fallback to LaMini ---
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
llm = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=250,
    temperature=0.2,
)

print("✅ Model loaded successfully: MBZUAI/LaMini-Flan-T5-248M")


# --- Tools ---
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
    prompt = (
        f"You are a telecom network expert analyzing call drop logs.\n"
        f"Region: {region or 'Unknown'}.\n"
        "Summarize the insights in the following structure:\n"
        "Region: (detected region)\n"
        "Observation: (summarize any trend or increase in call drops)\n"
        "Root Cause: (explain main causes using signal, congestion, and handoff data)\n"
        "Short Summary: (one concise statement)\n\n"
        "Logs:\n"
    )
    for s in snippets:
        prompt += f"- {s}\n"
    prompt += "\nRespond in 3-4 sentences maximum."
    response = llm(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    return response.strip()


# --- AI Recommendations ---
def generate_ai_recommendations(region: str, metrics: Dict) -> str:
    """
    Improved version: ensures the model always produces 3 distinct
    and context-aware technical recommendations.
    Works well with flan-t5-large and flan-t5-xl.
    """
    prompt = (
        f"You are an experienced telecom network engineer.\n"
        f"Analyze the following metrics for {region} and suggest three precise actions to reduce call drops.\n\n"
        f"Network Metrics:\n"
        f"- Average Signal Strength: {metrics.get('avg_signal')} dBm\n"
        f"- Congestion Level: {metrics.get('congestion_level')}\n"
        f"- Handoff Failure Rate: {metrics.get('handoff_pct')}%\n"
        f"- Dropout Rate: {metrics.get('drop_rate')}%\n\n"
        "Instructions:\n"
        "1. Suggest exactly three technical actions.\n"
        "2. Focus on signal, congestion, and handoff optimization.\n"
        "3. Each line must begin with a number (1., 2., 3.) and include:\n"
        "   (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "4. Do not add explanations outside the list.\n\n"
        "Example:\n"
        "1. Deploy microcells in dense areas - Priority: High - To reduce congestion during peak hours.\n"
        "2. Optimize antenna tilt and transmit power - Priority: Medium - To improve weak signal zones.\n"
        "3. Adjust handoff timers - Priority: High - To minimize call drops during mobility.\n\n"
        "Now write your final 3 recommendations:\n"
    )

    response = llm(prompt, max_new_tokens=200, do_sample=False, temperature=0.3)[0]["generated_text"]

    # --- Clean and extract numbered suggestions ---
    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    recs = []
    for ln in lines:
        if ln.startswith(("1.", "2.", "3.")):
            recs.append(ln)
        elif any(k in ln.lower() for k in ["priority", "optimize", "deploy", "handoff", "signal", "congestion"]):
            recs.append(f"{len(recs)+1}. {ln}")

    # --- Fallbacks if fewer than 3 ---
    default_recs = [
        "1. Deploy additional microcells or small cells during peak hours - Priority: High - To improve coverage and reduce congestion.",
        "2. Fine-tune handoff thresholds and reconfiguration timers - Priority: Medium - To reduce failed handovers.",
        "3. Increase backhaul capacity and monitor throughput - Priority: Medium - To improve data flow efficiency.",
    ]

    if len(recs) < 3:
        recs = default_recs[:3]
    elif len(recs) > 3:
        recs = recs[:3]

    return "\n".join(recs)




# --- Main Analysis ---
def analyze_region_query(user_query: str, region: str = None):
    hits = query_vector_db(user_query if not region else f"Region: {region}")
    # Filter relevant region results
    snippets = [
        h["document"]
        for h in hits
        if not region or h["metadata"].get("Region", "").lower() == region.lower()
    ]

    summary_txt = generate_ai_summary(snippets, region=region)

    # Extract metrics
    if hits:
        top_meta = hits[0]["metadata"]
        call_drops = float(top_meta.get("Call_Drops", 0))
        total_calls = 1000  # assumed base
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
