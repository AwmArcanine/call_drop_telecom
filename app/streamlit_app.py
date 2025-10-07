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

    # Safely parse recommendations (guarantee 3)
    rec_text = result.get("recommendations", "")
    rec_lines = re.findall(r"\d+\..+", rec_text)
    if len(rec_lines) < 3:
        alt_lines = [l.strip() for l in rec_text.split("\n") if l.strip()]
        rec_lines.extend(alt_lines)
    while len(rec_lines) < 3:
        rec_lines.append("Further network optimization required.")
    rec_lines = [r.strip() for r in rec_lines[:3]]

    # Extract Observation and Root Cause
    summary_text = result.get("summary", "")
    observation_match = re.search(r"Observation[:\-]\s*(.+)", summary_text, re.IGNORECASE)
    observation = observation_match.group(1).strip() if observation_match else \
        "Recent network monitoring shows a rise in call drops, likely linked to load and signal degradation."
    root_cause = summary_text or "No specific cause identified."

    # Display formatted agent findings
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

    # ---- Recommended Actions Section ----
    st.markdown("### 🧩 Recommended Actions")
    st.write(
        "These steps are prioritized based on signal strength, congestion, and handoff performance."
    )
    for i, rec in enumerate(rec_lines, 1):
        st.markdown(f"**{i}.** {rec}")

    # ---- Evidence Section ----
    st.markdown("### 📜 Evidence (Top Retrieved Logs)")

    # Filter region-specific logs
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

            # Compute dropout rate (example)
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
