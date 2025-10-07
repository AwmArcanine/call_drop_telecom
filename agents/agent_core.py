# agents/agent_core.py
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# --- Configurable ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "telecom_logs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_NAME = "MBZUAI/LaMini-Flan-T5-248M"  # Faster than flan-t5-base but still smart
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

# --- Initialize components ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

# --- Tools ---
def query_vector_db(query_text: str, top_k: int = TOP_K) -> List[Dict]:
    q_emb = embedder.encode([query_text], convert_to_numpy=True)[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "documents", "distances"])
    items = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        items.append({"document": doc, "metadata": meta, "distance": float(dist)})
    return items

def generate_ai_summary(snippets: List[str], region: str = None) -> str:
    prompt = (
        f"You are a telecom network expert analyzing call drop logs.\n"
        f"Region: {region or 'Unknown'}.\n"
        "Summarize the causes of call drops, referencing signal strength, congestion, and handoff failures.\n\n"
        "Here are the log entries:\n"
    )
    for s in snippets:
        prompt += f"- {s}\n"
    prompt += (
        "\nRespond in 2-3 sentences with: Region, Observation, Root Cause, and Short Summary."
    )
    response = llm(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    return response

def generate_ai_recommendations(region: str, metrics: Dict) -> str:
    prompt = (
        f"You are a telecom optimization assistant. Analyze the network conditions for region {region}.\n"
        f"Metrics:\n"
        f"- Average Signal Strength: {metrics.get('avg_signal')} dBm\n"
        f"- Congestion Level: {metrics.get('congestion_level')}\n"
        f"- Handoff Failure Rate: {metrics.get('handoff_pct')}%\n"
        f"- Dropout Rate: {metrics.get('drop_rate')}%\n\n"
        "Based on these, suggest **exactly three technical resolutions** to reduce call drops.\n"
        "Each suggestion must start with a number (1., 2., 3.) and include:\n"
        "- Specific action (e.g., deploy microcells, adjust handoff thresholds)\n"
        "- Priority (High/Medium/Low)\n"
        "- One-line rationale.\n\n"
        "Output Format:\n"
        "1. (Action) - Priority: (High/Medium/Low) - (Short reason)\n"
        "2. ...\n"
        "3. ..."
    )

    rec = llm(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]
    return rec


# --- Main Analysis ---
def analyze_region_query(user_query: str, region: str = None):
    hits = query_vector_db(user_query if not region else f"Region: {region}")
    snippets = [h["document"] for h in hits if not region or h["metadata"].get("Region", "").lower() == region.lower()]

    summary_txt = generate_ai_summary(snippets, region=region)

    if hits:
        top_meta = hits[0]["metadata"]
        call_drops = float(top_meta.get("Call_Drops", 0))
        total_calls = 1000  # assume normalized base if not available
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
