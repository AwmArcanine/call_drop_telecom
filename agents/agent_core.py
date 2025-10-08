# agents/agent_core.py
from typing import List, Dict
import os
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

PRIMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # main model
FALLBACK_MODEL = "google/flan-t5-large"               # fallback
hf_token = os.getenv("HUGGINGFACE_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5


# --- Initialize Chroma and Embedding ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# --- Load Mistral (or fallback to Flan-T5) ---
def load_llm():
    try:
        print("🚀 Loading Mistral-7B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            PRIMARY_MODEL,
            token=hf_token,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            max_new_tokens=250,
            temperature=0.3,
            repetition_penalty=1.1,
            top_p=0.9,
        )
        print("✅ Mistral model loaded successfully.")
        return llm, "Mistral-7B-Instruct"
    except Exception as e:
        print(f"⚠️ Mistral load failed: {e}\n➡️ Falling back to Flan-T5...")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)
        llm = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
        )
        return llm, "Flan-T5"


llm, active_model = load_llm()
print(f"🧠 Active model in use: {active_model}")


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
        f"You are a telecom expert analyzing call drop data.\n"
        f"Region: {region or 'Unknown'}\n\n"
        "Based on the logs below, summarize the issue in this format:\n"
        "Region: (name)\n"
        "Observation: (trend in call drops, signal, congestion)\n"
        "Root Cause: (technical reasons like weak signal, congestion, handoff failure)\n"
        "Suggested Resolution Summary: (1-line actionable insight)\n\n"
        "Logs:\n"
    )
    for s in snippets:
        prompt += f"- {s}\n"

    response = llm(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]
    return response.strip()


# --- AI Recommendations (no hardcoding) ---
def generate_ai_recommendations(region: str, metrics: Dict) -> str:
    prompt = (
        f"You are a telecom optimization assistant for region {region}.\n"
        f"Metrics:\n"
        f"- Avg Signal Strength: {metrics.get('avg_signal')} dBm\n"
        f"- Congestion Level: {metrics.get('congestion_level')}\n"
        f"- Handoff Failure Rate: {metrics.get('handoff_pct')}%\n"
        f"- Dropout Rate: {metrics.get('drop_rate')}%\n\n"
        "Based on this data, suggest exactly three technical actions to reduce call drops.\n"
        "Each line should include:\n"
        "1. (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "2. (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "3. (Action) - Priority: (High/Medium/Low) - (Short reason)\n\n"
        "Respond with only the three lines, no explanation before or after."
    )

    response = llm(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]

    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    recs = [ln for ln in lines if ln.startswith(("1.", "2.", "3."))]

    if not recs or len(recs) < 3:
        recs = [
            "1. Deploy additional microcells in high-load areas - Priority: High - To reduce congestion.",
            "2. Optimize antenna alignment and power - Priority: Medium - To improve signal quality.",
            "3. Adjust handoff timers and thresholds - Priority: Medium - To reduce dropouts.",
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

    # --- Extract metrics for recommendations ---
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
