from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("")
def chat(request: ChatRequest):
    return {
        "answer": "RAG pipeline not connected yet. This is a placeholder response.",
        "query_received": request.message,
    }