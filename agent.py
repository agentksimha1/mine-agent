import os
import pathlib
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pickle

# ---------------------------
# Setup
# ---------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Embeddings model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Load FAISS index + metadata
# ---------------------------
faiss_index_path = VECTORSTORE_PATH / "index.faiss"
faiss_meta_path = VECTORSTORE_PATH / "index.pkl"

if not faiss_index_path.exists():
    raise FileNotFoundError("❌ FAISS index not found. Run the vectorstore builder first.")

# Load FAISS index
index = faiss.read_index(str(faiss_index_path))

# ------ FIXED, PERMANENT METADATA LOADING ------
with open(faiss_meta_path, "rb") as f:
    meta = pickle.load(f)

documents = None

# 1) Case: meta is a list
if isinstance(meta, list):
    documents = meta

# 2) Case: meta is a tuple → scan inside
elif isinstance(meta, tuple):
    for item in meta:
        if isinstance(item, list):
            documents = item
            break
        if isinstance(item, dict) and "documents" in item:
            documents = item["documents"]
            break

# 3) Case: meta is a dict with "documents" key
elif isinstance(meta, dict):
    if "documents" in meta:
        documents = meta["documents"]

# 4) Deep search fallback
if documents is None:
    def find_docs(obj):
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                found = find_docs(v)
                if found:
                    return found
        if isinstance(obj, tuple):
            for v in obj:
                found = find_docs(v)
                if found:
                    return found
        return None
    documents = find_docs(meta)

# 5) If still None → raise error
if documents is None:
    raise TypeError("❌ Could not locate any list of documents inside index.pkl. "
                    "Rebuild the vectorstore.")

print(f"✅ Loaded {len(documents)} documents from index.pkl")

# ---------------------------
# Utility: Embed a query
# ---------------------------
def embed(text: str):
    return np.array(embed_model.encode([text]), dtype=np.float32)

# ---------------------------
# RAG Search
# ---------------------------
def search_faiss(query: str, k=5):
    q_emb = embed(query)
    distances, indices = index.search(q_emb, k)
    hits = [documents[i] for i in indices[0] if i != -1]
    return hits

# ---------------------------
# Generate Final Answer (Gemini)
# ---------------------------
def generate_answer(query: str, context_docs: list):
    context = "\n\n".join(context_docs)

    prompt = f"""
You are an expert mining assistant.

User question:
{query}

Relevant mining documents:
{context}

Answer concisely, factually, and directly.
"""

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text

# ---------------------------
# Full RAG Pipeline
# ---------------------------
def ask(query: str):
    context_docs = search_faiss(query, k=5)
    answer = generate_answer(query, context_docs)
    return answer

# ---------------------------
# CLI testing
# ---------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk me anything about mining: ")
        print("\n🔍 Searching FAISS...")
        print("💬 Answer:", ask(q))

