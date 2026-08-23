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


@app.get("/health")
def health():
    return {"status": "ok"}