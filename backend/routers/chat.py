import sys
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

RAG_DIR = Path(__file__).resolve().parent.parent.parent / "RAG"
sys.path.append(str(RAG_DIR))

from main import run_rag_pipeline  

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("")
def chat(request: ChatRequest):
    answer = run_rag_pipeline(request.message, verbose=False)
    return {"answer": answer}