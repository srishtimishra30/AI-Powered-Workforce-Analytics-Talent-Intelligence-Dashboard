import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

RAG_DIR = Path(__file__).resolve().parent.parent.parent / "RAG"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

spec = importlib.util.spec_from_file_location("rag_main", RAG_DIR / "main.py")
rag_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_main)
run_rag_pipeline = rag_main.run_rag_pipeline  

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("")
def chat(request: ChatRequest):
    answer = run_rag_pipeline(request.message, verbose=False)
    return {"answer": answer}