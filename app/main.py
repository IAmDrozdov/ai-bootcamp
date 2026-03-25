from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.dependencies import load_default_rubrics
from app.web.lessons import router as lessons_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_default_rubrics()
    yield


app = FastAPI(
    title="AI Student Assessment System",
    description="LLM-powered student work assessment — GenAI learning project",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(api_router)
app.include_router(lessons_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
