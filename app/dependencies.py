# FastAPI dependency injection for LLM and services.
#
# FastAPI's Depends() system wires up shared resources (config, LLM, chains)
# without global state. Each dependency is a function that returns a resource.
#
# lru_cache on get_settings ensures one Settings instance per process.
# LLM and chain are created per-request here for simplicity — in production
# you'd cache them too, but for learning it's clearer to see the creation flow.

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable

from app.chains.assessment_chain import build_assessment_chain
from app.config import Settings
from app.schemas.rubric import Rubric
from app.services.llm import create_llm

_rubrics_store: dict[str, Rubric] = {}


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_llm() -> ChatAnthropic:
    return create_llm(get_settings())


@lru_cache
def get_assessment_chain() -> Runnable:
    return build_assessment_chain(get_llm())


def get_rubrics_store() -> dict[str, Rubric]:
    return _rubrics_store


def load_default_rubrics() -> None:
    rubrics_dir = Path("data/rubrics")
    if not rubrics_dir.exists():
        return
    for path in rubrics_dir.glob("*.json"):
        data = json.loads(path.read_text())
        rubric = Rubric.model_validate(data)
        _rubrics_store[rubric.id] = rubric


SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMDep = Annotated[ChatAnthropic, Depends(get_llm)]
ChainDep = Annotated[Runnable, Depends(get_assessment_chain)]
RubricStoreDep = Annotated[dict[str, Rubric], Depends(get_rubrics_store)]
