from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.inference import TextToSQL


class InferenceRequest(BaseModel):
    question: str
    columns: list[str]
    mode: str = "greedy"
    beam_size: int = 4


app = FastAPI(title="Text-to-SQL Transformer")
frontend = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend), name="static")
inference = TextToSQL()


@app.get("/")
def index():
    return FileResponse(frontend / "index.html")


@app.post("/api/generate")
def generate(request: InferenceRequest):
    return inference.predict(
        request.question,
        request.columns,
        request.mode,
        request.beam_size,
    )