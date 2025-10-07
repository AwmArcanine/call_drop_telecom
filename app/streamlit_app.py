import re
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

            # Safely extract recommendations (1., 2., 3.) from model output
            rec_text = result.get("recommendations", "")
            rec_lines = re.findall(r"\d+\..+", rec_text)

            # Fallback handling — guarantee at least 3 items
            if len(rec_lines) < 3:
                # Try splitting by newline if regex finds fewer
                alt_lines = [l.strip() for l in rec_text.split("\n") if l.strip()]
                rec_lines.extend(alt_lines)

            # Final fallback defaults
            while len(rec_lines) < 3:
                rec_lines.append("Further network optimization required.")

            # Clean and keep only first 3
            rec_lines = [r.strip() for r in rec_lines[:3]]

            # Display formatted structured output
            st.markdown(
                f"""
                <div class="card">
                    <p><b>Region:</b> {region}</p>
                    <p><b>Observation:</b> (Summarized automatically by the AI model)</p>
                    <p><b>Root Cause:</b> {result['summary']}</p>
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

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

else:
    st.info("👈 Enter parameters and click **Analyze** to start the AI-powered analysis.")
