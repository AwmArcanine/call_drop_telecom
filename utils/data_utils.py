# utils/data_utils.py
import pandas as pd
import re
from typing import List, Dict

def load_telecom_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

def normalize_signal_strength(val: str) -> float:
    # convert like "-95 dBm" to -95.0
    if pd.isna(val):
        return None
    m = re.search(r"(-?\d+(\.\d+)?)", str(val))
    return float(m.group(1)) if m else None

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal_Str_dBm'] = df['Signal_Strength'].apply(normalize_signal_strength)
    df['Handoff_Failure_pct'] = df['Handoff_Failure'].astype(str).str.rstrip('%').astype(float)
    # create a textual 'document' for RAG
    def make_doc(row):
        return (
            f"Region: {row.Region} | Tower: {row.Tower_ID} | Date: {row.Date.date()} | "
            f"Call_Drops: {row.Call_Drops} | Signal: {row.Signal_Strength} | "
            f"Congestion: {row.Congestion_Level} | HandoffFailure: {row.Handoff_Failure} | Notes: {row.Notes}"
        )
    df['doc'] = df.apply(make_doc, axis=1)
    return df
