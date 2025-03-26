import streamlit as st
import requests

API_URL = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Teacher-Student Chat")

if prompt := st.chat_input("Message the student..."):
    st.session_state.chat_history.append({"role": "user", "content": f"Teacher: {prompt}"})
    response = requests.post(f"{API_URL}/student-response", json={
        "user_input": prompt,
        "chat_history": st.session_state.chat_history
    }).json()["response"]
    st.session_state.chat_history.append({"role": "assistant", "content": response})

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])