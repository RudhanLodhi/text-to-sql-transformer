from pathlib import Path
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import Request
from app.inference import TextToSQL


class InferenceRequest(BaseModel):
    question: str
    columns: list[str]
    mode: str = "greedy"
    beam_size: int = 4


app = FastAPI(title="Text-to-SQL Transformer")
app_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=app_dir / "templates")
app.mount(
    "/static",
    StaticFiles(directory=app_dir / "static"),
    name="static",
)
inference = TextToSQL()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.post("/api/generate")
def generate(request: InferenceRequest):
    return inference.predict(
        request.question,
        request.columns,
        request.mode,
        request.beam_size,
    )