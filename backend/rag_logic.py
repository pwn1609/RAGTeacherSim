import pickle
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import ollama
 
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open("textbook_passages.pkl", "rb") as f:
    textbook_passages = pickle.load(f)
index = faiss.read_index("vectorized_textbooks.faiss")

# Load scenarios data
def load_scenarios():
    try:
        scenarios_path = os.path.join("data", "scenarios.json")
        with open(scenarios_path, "r") as f:
            scenarios_data = json.load(f)
        return {s["scenario_id"]: s for s in scenarios_data}
    except Exception as e:
        print(f"Error loading scenarios: {e}")
        return {}

scenarios_dict = load_scenarios()


def generate_student_response(user_input, chat_history, scenario_id=None):
    # Get scenario context if available
    scenario_context = ""
    if scenario_id and scenario_id in scenarios_dict:
        scenario = scenarios_dict[scenario_id]
        scenario_type = scenario.get("type", "")
        
        if scenario_type == "Student Emotion" and "emotion" in scenario:
            scenario_context = f"You are feeling {scenario['emotion'].lower()}. "
        
        scenario_context += f"This is about: {scenario['description']}"

    # Create system prompt with scenario context
    system_prompt = {
        "role": "system",
        "content": f"You are an enthusiastic 2nd grade student who responds simply, sometimes distractedly. {scenario_context} You always respond as a second grader, and never as the teacher."
    }
    
    messages = [system_prompt] + chat_history + [{"role": "user", "content": f"Teacher: {user_input}"}]
    response = ollama.chat(model="llama3.2", messages=messages)
    reply = response.get("message", {}).get("content", "No response found.")
    return reply

def retrieve_textbook_context(query, top_k=3):
    query_embedding = np.array(embedder.encode([query])).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [textbook_passages[i] for i in indices[0] if i < len(textbook_passages)]

def generate_expert_advice(question, conversation_history, scenario_id=None):
    # Create transcript from conversation history
    transcript = "\n".join(
        f"{'Student' if m['role'] == 'assistant' else 'Teacher'}: {m['content']}"
        for m in conversation_history
    )
    
    # Add scenario context if available
    scenario_context = ""
    if scenario_id and scenario_id in scenarios_dict:
        scenario = scenarios_dict[scenario_id]
        scenario_context = f"Scenario: {scenario['title']}\nDescription: {scenario['description']}\n\n"
    
    # Augment query with scenario information for better retrieval
    retrieval_query = question
    if scenario_id and scenario_id in scenarios_dict:
        scenario = scenarios_dict[scenario_id]
        retrieval_query = f"{question} {scenario['title']} {scenario['description']}"
    
    # Retrieve relevant textbook passages
    passages = retrieve_textbook_context(retrieval_query)
    
    # Create prompt with all context
    prompt = {
        "role": "system",
        "content": "You are an expert teacher trainer who provides specific, actionable advice based on educational best practices and textbook knowledge. Keep your answers focused on practical strategies for the specific scenario."
    }
    
    user_input = {
        "role": "user",
        "content": f"{scenario_context}Question: {question}\n\nConversation:\n{transcript}\n\nRelevant Teaching Principles:\n" + "\n".join(f"- {p}" for p in passages)
    }
    
    messages = [prompt, user_input]
    response = ollama.chat(model="llama3.2", messages=messages)
    return response.get("message", {}).get("content", "No response found.")

