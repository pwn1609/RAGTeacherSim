from fastapi import FastAPI
from pydantic import BaseModel
from .rag_logic import generate_student_response, generate_expert_advice

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str
    chat_history: list[dict]

class AdviceRequest(BaseModel):
    question: str
    conversation_history: list[dict]

@app.post("/student-response")
def student_response(req: ChatRequest):
    response = generate_student_response(req.user_input, req.chat_history)
    return {"response": response}

@app.post("/expert-advice")
def expert_advice(req: AdviceRequest):
    response = generate_expert_advice(req.question, req.conversation_history)
    return {"response": response}