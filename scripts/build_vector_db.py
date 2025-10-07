# scripts/build_vector_db.py
import argparse
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pandas as pd
from utils.data_utils import load_telecom_csv, preprocess_df
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def build_chroma(csv_path, chroma_dir, model_name="all-MiniLM-L6-v2", collection_name="telecom_logs"):
    print("Loading CSV...")
    df = load_telecom_csv(csv_path)
    df = preprocess_df(df)
    docs = df['doc'].tolist()
    df['Date'] = df['Date'].astype(str)
    metadatas = df.to_dict(orient='records')
    ids = [f"row_{i}" for i in range(len(docs))]

    print("Creating embedding model...")
    embedder = SentenceTransformer(model_name)

    print("Building embeddings (this may take a while)...")
    embeddings = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    print(f"Starting Chroma client at dir {chroma_dir}...")
    client = chromadb.PersistentClient(path=chroma_dir)
    # create or get collection
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(name=collection_name)

    print("Upserting vectors into Chroma collection...")
    print("✅ Sample metadata:", metadatas[0])
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    print("Chroma DB build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to telecom_logs.csv")
    parser.add_argument("--chroma_dir", default="chroma_db", help="Chroma DB directory")
    args = parser.parse_args()
    os.makedirs(args.chroma_dir, exist_ok=True)
    build_chroma(args.csv, args.chroma_dir)
