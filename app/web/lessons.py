from pathlib import Path

import markdown
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

LESSONS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "lessons"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

md = markdown.Markdown(
    extensions=["fenced_code", "codehilite", "tables", "toc", "nl2br"],
    extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": True},
    },
)

TITLE_MAP = {
    "topic_01_prompt_engineering": "Промпт-инжиниринг",
    "topic_02_langchain_lcel": "LangChain Core + LCEL",
    "topic_03_structured_output": "Structured Output",
    "topic_04_streaming": "Streaming",
    "topic_05_rag": "RAG",
    "topic_06_langgraph_agents": "LangGraph & Agents",
    "topic_07_conversational_ai": "Conversational AI",
    "topic_08_observability": "Observability",
    "topic_09_evaluation": "Evaluation",
    "topic_10_production_patterns": "Production Patterns",
}


def _get_sorted_topics() -> list[dict]:
    files = sorted(LESSONS_DIR.glob("topic_*.md"))
    topics = []
    for i, f in enumerate(files, 1):
        slug = f.stem
        topics.append({
            "number": i,
            "slug": slug,
            "title": TITLE_MAP.get(slug, slug.replace("_", " ").title()),
            "path": f,
        })
    return topics


@router.get("/", response_class=HTMLResponse)
async def topics_list(request: Request):
    topics = _get_sorted_topics()
    return templates.TemplateResponse(request, "topics.html", {"topics": topics})


@router.get("/lessons/{slug}", response_class=HTMLResponse)
async def topic_detail(request: Request, slug: str):
    topics = _get_sorted_topics()
    slugs = [t["slug"] for t in topics]

    if slug not in slugs:
        raise HTTPException(status_code=404, detail="Lesson not found")

    idx = slugs.index(slug)
    topic = topics[idx]

    md.reset()
    content = topic["path"].read_text(encoding="utf-8")
    html_content = md.convert(content)

    prev_slug = slugs[idx - 1] if idx > 0 else None
    next_slug = slugs[idx + 1] if idx < len(slugs) - 1 else None

    return templates.TemplateResponse(request, "topic_detail.html", {
        "title": topic["title"],
        "content": html_content,
        "prev_slug": prev_slug,
        "next_slug": next_slug,
    })
