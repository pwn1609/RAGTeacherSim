from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .rag_logic import generate_student_response, generate_expert_advice

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str
    chat_history: list[dict]
    scenario_id: Optional[str] = None

class AdviceRequest(BaseModel):
    question: str
    conversation_history: list[dict]
    scenario_id: Optional[str] = None

@app.post("/student-response")
def student_response(req: ChatRequest):
    response = generate_student_response(req.user_input, req.chat_history, req.scenario_id)
    return {"response": response}

@app.post("/expert-advice")
def expert_advice(req: AdviceRequest):
    response = generate_expert_advice(req.question, req.conversation_history, req.scenario_id)
    return {"response": response}