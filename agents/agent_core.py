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
DEVICE = "cuda"
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
    torch_dtype=torch.float16,
    temperature=0.2,
    device_map="auto"
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
    Final fixed version for FLAN-T5 models — produces 3 distinct and accurate
    recommendations with stable decoding and fallback handling.
    """
    prompt = (
        f"You are a telecom optimization assistant analyzing call drop causes in {region}.\n"
        f"Metrics:\n"
        f"Signal Strength: {metrics.get('avg_signal')} dBm\n"
        f"Congestion Level: {metrics.get('congestion_level')}\n"
        f"Handoff Failure: {metrics.get('handoff_pct')}%\n"
        f"Dropout Rate: {metrics.get('drop_rate')}%\n\n"
        "Generate exactly 3 unique and actionable resolutions to reduce call drops.\n"
        "Each suggestion must follow this format:\n"
        "1. (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "2. (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "3. (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "Keep it concise and data-driven.\n"
    )

    # Generate response using deterministic decoding
    response = llm(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        repetition_penalty=1.2,
    )[0]["generated_text"]

    # --- Parse output cleanly ---
    recs = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith(("1.", "2.", "3.")):
            recs.append(line)
        elif len(recs) < 3 and any(word in line.lower() for word in ["optimize", "deploy", "handoff", "antenna", "backhaul", "signal"]):
            recs.append(f"{len(recs)+1}. {line}")

    # --- Hard fallback if model misses structure ---
    if len(recs) < 3:
        recs = [
            "1. Optimize antenna tilt and power settings - Priority: High - To improve weak signal coverage.",
            "2. Deploy microcells in dense areas - Priority: High - To reduce congestion during peak hours.",
            "3. Fine-tune handoff thresholds - Priority: Medium - To prevent dropped calls during mobility.",
        ]

    return "\n".join(recs[:3])


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
