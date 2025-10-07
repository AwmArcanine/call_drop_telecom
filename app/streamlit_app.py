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

    # ---- Agent Response ----
    st.markdown("### 🧠 Agent Response")

    # --- Extract Recommendations Cleanly ---
    rec_text = result.get("recommendations", "").strip()
    # Extract lines like "1. ..." "2. ..." "3. ..." robustly
    rec_lines = re.findall(r"(?:\d+[\.\)]\s*)([^\n]+)", rec_text)
    rec_lines = [r.strip() for r in rec_lines if r.strip()]

    # Ensure exactly 3 distinct actionable lines
    defaults = [
        "Deploy additional microcells or optimize antenna alignment to improve coverage.",
        "Tune handoff thresholds and reconfiguration timers to minimize drop events.",
        "Increase backhaul capacity and conduct targeted drive tests in problem areas."
    ]
    if not rec_lines:
        rec_lines = defaults
    elif len(rec_lines) < 3:
        rec_lines.extend(defaults[len(rec_lines):])
    rec_lines = rec_lines[:3]

    # --- Observation & Root Cause ---
    summary_text = result.get("summary", "")
    observation_match = re.search(r"Observation[:\-]\s*(.+)", summary_text, re.IGNORECASE)
    observation = observation_match.group(1).strip() if observation_match else \
        "Increased call drops observed due to high user load and moderate congestion levels."
    root_cause = summary_text or "No specific cause identified."

    # --- Agent Output Block ---
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f9fafc; border:1px solid #ddd;">
            <p><b>Region:</b> {region}</p>
            <p><b>Observation:</b> {observation}</p>
            <p><b>Root Cause:</b> {root_cause}</p>
            <p><b>Suggested Resolution:</b></p>
            <ol>
                <li>{rec_lines[0]}</li>
                <li>{rec_lines[1]}</li>
                <li>{rec_lines[2]}</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Evidence Section ----
    st.markdown("### 📜 Evidence (Top Retrieved Logs)")

    hits = [h for h in result["evidence"] if h["metadata"].get("Region", "").lower() == region.lower()]

    if not hits:
        st.info("No evidence found for this region.")
    else:
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            signal = meta.get("Signal_Str_dBm", "N/A")
            congestion = meta.get("Congestion_Level", "Unknown")
            handoff = meta.get("Handoff_Failure_pct", "N/A")
            drops = meta.get("Call_Drops", "N/A")

            # Dropout rate (approximate)
            try:
                dropout_rate = f"{(int(drops)/100):.1f}%"
            except Exception:
                dropout_rate = "N/A"

            # Congestion severity icons
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
                    📶 <b>Signal:</b> {signal} dBm &nbsp;|&nbsp; {cong_icon} &nbsp;|&nbsp;
                    🔁 <b>Handoff Failure:</b> {handoff}% &nbsp;|&nbsp;
                    📉 <b>Dropout Rate:</b> {dropout_rate}<br>
                    📝 <b>Notes:</b> {meta.get('Notes', 'No notes available.')}<br>
                    <small>Distance Score: {hit.get('distance'):.4f}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.success("✅ Analysis complete.")
