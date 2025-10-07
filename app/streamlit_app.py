import sys, os, re, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agents.agent_core import analyze_region_query

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="Call Drop Analyzer", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.header("🔍 Query Parameters")
    region = st.text_input("Region (optional)", value="Hyderabad")
    tower = st.text_input("Tower ID (optional)", value="")
    user_q = st.text_area("Query", value=f"Why are call drops high in {region}?")
    run = st.button("🚀 Analyze")

st.title("📊 Agent: Call Drop Analysis")
st.markdown(
    "This AI Agent analyzes telecom network logs to explain **call drop causes**, detect **patterns**, "
    "and suggest **data-driven resolutions** automatically."
)

# ---- Cached Analyzer ----
@st.cache_data(show_spinner=False)
def cached_analyze(user_q, region):
    """Cached version to speed up repeated queries."""
    return analyze_region_query(user_q, region=region)

# ---- Run Agent ----
if run:
    with st.spinner("⏳ Analyzing network logs..."):
        progress = st.progress(0)
        for i in range(50):
            time.sleep(0.02)
            progress.progress(i + 1)
        result = cached_analyze(user_q, region if region else None)
        for i in range(50, 100):
            time.sleep(0.02)
            progress.progress(i + 1)
        progress.empty()

    # ---- Extract relevant data ----
    hits = [h for h in result["evidence"] if h["metadata"].get("Region", "").lower() == region.lower()]
    if not hits:
        hits = result["evidence"][:1]  # fallback to first available hit

    # --- Use top hit for contextual info ---
    top = hits[0]["metadata"]
    tower_id = top.get("Tower_ID", "N/A")
    signal = top.get("Signal_Str_dBm", "N/A")
    congestion = top.get("Congestion_Level", "N/A")
    handoff = top.get("Handoff_Failure_pct", "N/A")
    drops = top.get("Call_Drops", "N/A")
    date = top.get("Date", "N/A")

    # Compute dropout rate safely
    try:
        dropout_rate = f"{(int(drops)/100):.1f}%"
    except Exception:
        dropout_rate = "N/A"

    # Generate Observation heuristically
    observation = f"Call drops increased to {drops} on {date}."
    if signal != "N/A":
        observation += f" Signal levels degraded around {signal} dBm."
    if "high" in str(congestion).lower():
        observation += " Congestion detected during peak hours."

    # Build Root Cause line
    root_cause = (
        f"Weak signal ({signal} dBm) + {congestion.capitalize()} congestion + {handoff}% handoff failure."
    )

    # Get recommendations and ensure 3 points
    rec_text = result.get("recommendations", "").strip()
    rec_lines = re.findall(r"\d+\..+", rec_text)
    if not rec_lines:
        rec_lines = re.split(r"[.;]\s+", rec_text)
    rec_lines = [r.strip(" -•\n") for r in rec_lines if len(r.strip()) > 5]
    while len(rec_lines) < 3:
        rec_lines.append("Further network optimization required.")
    rec_lines = rec_lines[:3]

    # ---- Display Agent Response ----
    st.markdown("### 🧠 Agent Response")
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f9fafc; border:1px solid #ddd;">
            <ul style="list-style-type:none; padding-left:0;">
                <li><b>Region:</b> {region}, Tower {tower_id}</li>
                <li><b>Observation:</b> {observation}</li>
                <li><b>Root Cause:</b> {root_cause}</li>
                <li><b>Suggested Resolution:</b></li>
                <ol>
                    <li>{rec_lines[0]}</li>
                    <li>{rec_lines[1]}</li>
                    <li>{rec_lines[2]}</li>
                </ol>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Evidence Section ----
    st.markdown("### 📜 Evidence (Top Retrieved Logs)")

    if not hits:
        st.info("No evidence found for this region.")
    else:
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            signal = meta.get("Signal_Str_dBm", "N/A")
            congestion = meta.get("Congestion_Level", "Unknown")
            handoff = meta.get("Handoff_Failure_pct", "N/A")
            drops = meta.get("Call_Drops", "N/A")
            notes = meta.get("Notes", "No notes available.")

            try:
                dropout_rate = f"{(int(drops)/100):.1f}%"
            except Exception:
                dropout_rate = "N/A"

            if "high" in str(congestion).lower():
                cong_icon = "🔴 High"
            elif "medium" in str(congestion).lower():
                cong_icon = "🟠 Medium"
            else:
                cong_icon = "🟢 Low"

            st.markdown(
                f"""
                <div style="padding:10px; margin-bottom:10px; border:1px solid #ddd; border-radius:10px;">
                    <b>Hit {i}: {meta.get('Region')} | {meta.get('Tower_ID')} | {meta.get('Date')}</b><br>
                    📶 <b>Signal:</b> {signal} dBm &nbsp;|&nbsp;
                    {cong_icon} &nbsp;|&nbsp;
                    🔁 <b>Handoff Failure:</b> {handoff}% &nbsp;|&nbsp;
                    📉 <b>Dropout Rate:</b> {dropout_rate}<br>
                    📝 <b>Notes:</b> {notes}<br>
                    <small>Distance Score: {hit.get('distance'):.4f}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.success("✅ Analysis complete.")
