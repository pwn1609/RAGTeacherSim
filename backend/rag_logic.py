import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
 
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open("C:\\Users\\pwn16\\OneDrive\\Documents\\School\\CS-339R\\RAGTeacherSim\\backend\\textbook_passages.pkl", "rb") as f:
    textbook_passages = pickle.load(f)
index = faiss.read_index("C:\\Users\\pwn16\\OneDrive\\Documents\\School\\CS-339R\\RAGTeacherSim\\backend\\vectorized_textbooks.faiss")

def check_appropriate_teacher_behavior(user_input):
    evaluation_prompt = {
        "role": "system",
        "content": """Evaluate whether the following teacher input is appropriate for a 2nd grade classroom.
        Inappropriate behavior includes:
        - Yelling or using ALL CAPS excessively
        - Using insulting or demeaning language
        - Making threats or using intimidation
        - Using inappropriate adult language or topics
        - Making personal comments unrelated to learning
        - Anything that would be considered verbal abuse
        
        Output ONLY \"inappropriate\" if the input is inappropriate, or \"appropriate\" if it is acceptable.
        Do not explain your reasoning - just return one word."""
    }
    response = ollama.chat(model="llama3.2", messages=[
        evaluation_prompt,
        {"role": "user", "content": f"Teacher's input: {user_input}"}
    ])
    result = response.get("message", {}).get("content", "").strip().lower()
    return "inappropriate" not in result

def generate_student_response(user_input, chat_history):
    if not check_appropriate_teacher_behavior(user_input):
        return "[That reply was not appropriate.]"

    chat_history.append({"role": "user", "content": f"Teacher: {user_input}"})
    system_prompt = {
        "role": "system",
        "content": "You are an enthusiastic 2nd grade student who responds simply, sometimes distractedly..."
    }
    messages = [system_prompt] + chat_history
    response = ollama.chat(model="llama3.2", messages=messages)
    reply = response.get("message", {}).get("content", "No response found.")
    chat_history.append({"role": "assistant", "content": reply})
    return reply

def retrieve_textbook_context(query, top_k=3):
    query_embedding = np.array(embedder.encode([query])).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [textbook_passages[i] for i in indices[0] if i < len(textbook_passages)]

def generate_expert_advice(question, conversation_history):
    transcript = "\n".join(
        f"{'Student' if m['role'] == 'assistant' else 'Teacher'}: {m['content']}"
        for m in conversation_history
    )
    passages = retrieve_textbook_context(transcript)
    prompt = {
        "role": "system",
        "content": "You are an expert teacher trainer..."
    }
    user_input = {
        "role": "user",
        "content": f"Question: {question}\n\nConversation:\n{transcript}\n\nTeaching Principles:\n" + "\n".join(f"- {p}" for p in passages)
    }
    messages = [prompt, user_input]
    response = ollama.chat(model="llama3.2", messages=messages)
    return response.get("message", {}).get("content", "No response found.")

