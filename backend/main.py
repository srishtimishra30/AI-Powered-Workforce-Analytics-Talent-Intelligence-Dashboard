import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import analytics, predictions, chat

app = FastAPI(title="Workforce Analytics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analytics.router)
app.include_router(predictions.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {
        "message": "Workforce Analytics Backend API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {"status": "ok"}