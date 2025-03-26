### BackEnd API Contract Documentation

#### POST `/student-response`
- **Description**: Generates a simulated student response based on teacher input and prior chat history.
- **Request Body**:
```json
{
  "user_input": "string",
  "chat_history": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```
- **Response**:
```json
{
  "response": "string"
}
```

---

#### POST `/expert-advice`
- **Description**: Returns advice from an expert teacher based on a question and the current conversation history.
- **Request Body**:
```json
{
  "question": "string",
  "conversation_history": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```
- **Response**:
```json
{
  "response": "string"
}
```

---

### 🚀 Backend Startup Instructions

1. **Install dependencies**:
```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu ollama
```

2. **Start the backend server**:
```bash
uvicorn backend.main:app --reload
```
- Runs at: http://localhost:8000
- Docs available at: http://localhost:8000/docs

---

### 🎨 Frontend Startup Instructions

1. **Install dependencies**:
```bash
pip install streamlit requests
```

2. **Run the Streamlit frontend**:
```bash
streamlit run ui/streamlit_app.py
```