import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agents.agent_core import analyze_region_query
import time

# --- Streamlit Config ---
st.set_page_config(page_title="AI Call Drop Analyzer", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; }
    .card {
        background-color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .section-title {
        font-weight: 700;
        font-size: 1.2rem;
        color: #1e3a8a;
        margin-top: 1.2rem;
    }
    .metric {
        font-weight: 600;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📡 Agent: Call Drop Analysis")
st.markdown(
    "This AI agent analyzes telecom network logs to identify **call drop causes**, detect **performance patterns**, "
    "and recommend **data-driven network optimizations**."
)

# --- Sidebar (Input Parameters) ---
with st.sidebar:
    st.header("🔍 Query Parameters")
    region = st.selectbox(
        "Select Region",
        ["Hyderabad", "Mumbai", "Kolkata", "Delhi", "Chennai"],
        index=0,
    )
    tower = st.text_input("Tower ID (optional)", value="")
    user_q = st.text_area(
        "Query",
        value=f"Why are call drops high in {region}?",
        height=100,
    )
    run = st.button("🚀 Analyze")

# --- Helper functions for color indicators ---
def get_congestion_color(level):
    if not level:
        return "⚪ Unknown"
    l = level.lower()
    if "low" in l: return "🟢 Low"
    elif "medium" in l: return "🟡 Medium"
    elif "high" in l: return "🔴 High"
    return "⚪ Unknown"

def get_signal_color(signal_strength):
    try:
        signal = float(signal_strength)
    except (TypeError, ValueError):
        return "⚪ Unknown"
    if signal >= -80:
        return "🟢 Strong"
    elif -95 <= signal < -80:
        return "🟡 Moderate"
    else:
        return "🔴 Weak"

# --- Main Processing ---
if run:
    progress_bar = st.progress(0, text="Initializing agent...")

    with st.spinner("🤖 Running AI analysis... please wait"):
        try:
            # Step 1: Retrieve vector data
            progress_bar.progress(25, text="📡 Retrieving telecom logs...")
            time.sleep(0.5)
            result = analyze_region_query(user_q, region=region)

            # Step 2: AI reasoning
            progress_bar.progress(60, text="🧠 Generating AI insights...")
            time.sleep(0.5)

            # Step 3: Done
            progress_bar.progress(100, text="✅ Analysis complete!")
            st.success("AI analysis completed successfully!")

            # --- Agent Response ---
            st.markdown("### 🧠 Agent Response")
            st.markdown(
                f"""
                <div class="card">
                    <p><b>Region:</b> {region}</p>
                    <p><b>Observation:</b> (Summarized automatically by the AI model)</p>
                    <p><b>Root Cause:</b> {result['summary']}</p>
                    <p><b>Suggested Resolution:</b></p>
                    <ol>
                        <li>{result['recommendations'].splitlines()[0]}</li>
                        <li>{result['recommendations'].splitlines()[1] if len(result['recommendations'].splitlines())>1 else ''}</li>
                        <li>{result['recommendations'].splitlines()[2] if len(result['recommendations'].splitlines())>2 else ''}</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Evidence Section ---
            st.markdown("### 📂 Evidence (Top Retrieved Logs)")
            region_hits = [
                h for h in result["evidence"]
                if h["metadata"].get("Region", "").lower() == region.lower()
            ]

            if not region_hits:
                st.warning(f"No relevant logs found for region: **{region}**.")
            else:
                for i, hit in enumerate(region_hits, 1):
                    meta = hit["metadata"]
                    signal_color = get_signal_color(meta.get("Signal_Str_dBm"))
                    congestion_color = get_congestion_color(meta.get("Congestion_Level"))

                    # Dropout rate calculation (per 1000 calls assumption)
                    call_drops = float(meta.get("Call_Drops", 0))
                    dropout_rate = round((call_drops / 1000) * 100, 2)

                    st.markdown(
                        f"""
                        <div class="card">
                            <b>Hit {i}:</b> {meta.get("Region")} | Tower: {meta.get("Tower_ID")} | Date: {meta.get("Date")}<br><br>
                            <span class="metric">📶 Signal:</span> {meta.get("Signal_Str_dBm")} dBm ({signal_color})<br>
                            <span class="metric">🚦 Congestion:</span> {meta.get("Congestion_Level")} ({congestion_color})<br>
                            <span class="metric">🔄 Handoff Failure:</span> {meta.get("Handoff_Failure_pct")}%<br>
                            <span class="metric">📉 Dropout Rate:</span> {dropout_rate}%<br>
                            <span class="metric">📝 Notes:</span> {meta.get("Notes")}<br>
                            <b>📊 Distance:</b> {hit.get("distance"):.4f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

else:
    st.info("👈 Enter parameters and click **Analyze** to start the AI-powered analysis.")
