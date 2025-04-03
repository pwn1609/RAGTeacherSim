# vectorize_textbooks.py
import os
import faiss
import numpy as np
import pickle
import openai # Use OpenAI library
import sys
from time import sleep

# --- Configuration ---
TEXTBOOKS_DIR = "CleanedTextBooks" # Directory containing your .txt textbook files
FAISS_INDEX_PATH = "vectorized_textbooks.faiss"
TEXTBOOK_PASSAGES_PATH = "textbook_passages.pkl"
# ** IMPORTANT: Use the same OpenAI embedding model here and in streamlit_app.py **
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # Recommended. Or "text-embedding-3-large", "text-embedding-ada-002"
CHUNK_SIZE = 500 # Approximate characters per chunk
BATCH_SIZE = 100 # Process passages in batches for API efficiency & rate limits

# --- Helper Functions ---

def load_textbooks(directory):
    """Reads all .txt files in the directory and returns a list of (filename, content)."""
    textbooks = []
    print(f"Looking for textbooks in: {os.path.abspath(directory)}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    found_files = False
    for filename in os.listdir(directory):
        if filename.lower().endswith(".txt"):
            found_files = True
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    print(f"  Loading: {filename}")
                    textbooks.append((filename, file.read()))
            except Exception as e:
                print(f"  Error reading {filename}: {e}", file=sys.stderr)
    if not found_files:
         print(f"Warning: No .txt files found in {directory}.", file=sys.stderr)
    return textbooks

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks based on sentences."""
    sentences = text.replace('\n', ' ').split('. ') # Simple sentence split, replace newlines
    sentences = [s.strip() for s in sentences if s.strip()] # Remove empty strings

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence_with_period = sentence + "." # Add period back
        if not current_chunk:
            current_chunk = sentence_with_period
        elif len(current_chunk) + len(sentence_with_period) + 1 <= chunk_size:
            current_chunk += " " + sentence_with_period
        else:
            # Chunk is full, add it and start a new one with overlap
            chunks.append(current_chunk)
            # Create overlap: find last few sentences of the previous chunk
            overlap_sentences = current_chunk.split('. ')[-3:] # Take last 3 sentences approx
            overlap_text = ". ".join(overlap_sentences)
            # Start new chunk with overlap + current sentence
            current_chunk = overlap_text + (" " if overlap_text else "") + sentence_with_period
            # Trim if overlap made it too long already (edge case)
            if len(current_chunk) > chunk_size * 1.2: # Allow some flexibility
                 current_chunk = sentence_with_period # Fallback to just the sentence

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk)

    print(f"    Split into {len(chunks)} chunks.")
    return chunks

# --- Main Processing ---

if __name__ == "__main__":
    print(f"--- Starting Textbook Vectorization using OpenAI ---")
    print(f"!!! This will use the OpenAI API and may incur costs !!!")

    # Set up OpenAI API key from Environment Variable
    try:
        # Best practice for scripts: use environment variables
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
             raise ValueError("OPENAI_API_KEY environment variable not set.\nPlease set it before running this script (e.g., export OPENAI_API_KEY='your_key')")
        print("OpenAI API key configured from environment variable.")
    except Exception as e:
        print(f"Error configuring OpenAI API key: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Load textbooks
    print(f"[1/5] Loading textbooks from '{TEXTBOOKS_DIR}'...")
    textbooks = load_textbooks(TEXTBOOKS_DIR)
    if not textbooks:
        print("No textbooks loaded. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded content from {len(textbooks)} files.")

    # Step 2: Process & chunk text
    print("\n[2/5] Chunking textbooks into passages...")
    all_passages = []
    for filename, content in textbooks:
        print(f"  Processing {filename}...")
        chunks = chunk_text(content, chunk_size=CHUNK_SIZE)
        all_passages.extend(chunks)
    print(f"Total passages created: {len(all_passages)}")
    if not all_passages:
        print("No passages created from textbooks. Check content and chunking logic. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Generate embeddings using OpenAI
    print(f"\n[3/5] Generating embeddings using OpenAI model: {OPENAI_EMBEDDING_MODEL}...")
    all_embeddings = []
    total_passages = len(all_passages)

    for i in range(0, total_passages, BATCH_SIZE):
        batch_passages = all_passages[i:min(i + BATCH_SIZE, total_passages)]
        current_batch_num = i // BATCH_SIZE + 1
        total_batches = (total_passages + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Processing batch {current_batch_num}/{total_batches} ({len(batch_passages)} passages)...")

        try:
            # Call OpenAI Embeddings API
            response = openai.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=batch_passages
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            sleep(0.5) # Add a small delay to respect potential rate limits

        except openai.APIError as e:
             print(f"  OpenAI API Error on batch {current_batch_num}: {e}. Retrying once after delay...", file=sys.stderr)
             sleep(5) # Wait longer before retry
             try:
                 response = openai.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch_passages)
                 batch_embeddings = [item.embedding for item in response.data]
                 all_embeddings.extend(batch_embeddings)
                 print(f"  Retry successful for batch {current_batch_num}.")
             except Exception as retry_e:
                 print(f"  Retry failed for batch {current_batch_num}: {retry_e}. Skipping batch.", file=sys.stderr)
                 all_embeddings.extend([None] * len(batch_passages)) # Add placeholders for failed batch

        except Exception as e:
            print(f"  Non-API Error generating embeddings for batch {current_batch_num}: {e}", file=sys.stderr)
            all_embeddings.extend([None] * len(batch_passages)) # Add placeholders

    # Filter out passages where embedding generation failed
    valid_embeddings_data = [(emb, passage) for emb, passage in zip(all_embeddings, all_passages) if emb is not None]
    num_failed = len(all_passages) - len(valid_embeddings_data)
    if num_failed > 0:
        print(f"Warning: Failed to generate embeddings for {num_failed} passages.", file=sys.stderr)
    if not valid_embeddings_data:
        print("Error: No embeddings were successfully generated. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Prepare final numpy array and corresponding passages list
    final_embeddings_np = np.array([item[0] for item in valid_embeddings_data]).astype('float32')
    final_passages = [item[1] for item in valid_embeddings_data]
    print(f"Embeddings generated successfully for {len(final_passages)} passages.")
    print(f"Output embeddings shape: {final_embeddings_np.shape}")

    # Step 4: Create FAISS Index
    print("\n[4/5] Creating FAISS index...")
    dimension = final_embeddings_np.shape[1] # Dimension determined by OpenAI model
    index = faiss.IndexFlatL2(dimension)
    print(f"  FAISS index type: IndexFlatL2, Dimension: {dimension}")
    print(f"  Adding {final_embeddings_np.shape[0]} vectors to the index...")
    try:
        index.add(final_embeddings_np)
        print(f"  Index size: {index.ntotal} vectors.")
    except Exception as e:
        print(f"Error adding vectors to FAISS index: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 5: Save FAISS Index & *Corresponding* Passages
    print("\n[5/5] Saving FAISS index and passages...")
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"  FAISS index saved to: '{FAISS_INDEX_PATH}'")
    except Exception as e:
        print(f"Error saving FAISS index: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Save only the passages for which embeddings were successfully created
        with open(TEXTBOOK_PASSAGES_PATH, "wb") as f:
            pickle.dump(final_passages, f)
        print(f"  Corresponding textbook passages saved to: '{TEXTBOOK_PASSAGES_PATH}'")
    except Exception as e:
        print(f"Error saving passages pickle file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Vectorization Complete ---")