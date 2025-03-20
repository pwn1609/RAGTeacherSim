import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

TEXTBOOKS_DIR = "./CleanedTextBooks/" 
FAISS_INDEX_PATH = "vectorized_textbooks.faiss" 
TEXTBOOK_PASSAGES_PATH = "textbook_passages.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

def load_textbooks(directory):
    """Reads all .txt files in the directory and returns a list of (filename, content)."""
    textbooks = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                textbooks.append((filename, file.read()))
    return textbooks

def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks of approximately `chunk_size` characters."""
    sentences = text.split(". ")  # Naive sentence split
    chunks, current_chunk = [], []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ------------- Main Processing ------------- #
# Step 1: Load textbooks
print("Loading textbooks...")
textbooks = load_textbooks(TEXTBOOKS_DIR)

# Step 2: Process & chunk text
print("Chunking textbooks into passages...")
all_passages = []
for filename, content in textbooks:
    chunks = chunk_text(content)
    all_passages.extend(chunks)

# Step 3: Generate embeddings
print(f"🔍 Generating embeddings using {EMBEDDING_MODEL}...")
embeddings = np.array(embedder.encode(all_passages), dtype=np.float32)

# Step 4: Create FAISS Index
print("📂 Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
index.add(embeddings)

# Step 5: Save FAISS Index & Passages
print("💾 Saving FAISS index and passages...")
faiss.write_index(index, FAISS_INDEX_PATH)

with open(TEXTBOOK_PASSAGES_PATH, "wb") as f:
    pickle.dump(all_passages, f)

print(f"Saved FAISS index to '{FAISS_INDEX_PATH}' and passages to '{TEXTBOOK_PASSAGES_PATH}'.")
