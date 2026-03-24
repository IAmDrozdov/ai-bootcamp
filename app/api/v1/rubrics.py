from fastapi import APIRouter, HTTPException

from app.dependencies import RubricStoreDep
from app.schemas.rubric import Rubric

router = APIRouter(prefix="/rubrics", tags=["rubrics"])


@router.get("")
async def list_rubrics(rubrics: RubricStoreDep) -> list[Rubric]:
    return list(rubrics.values())


@router.get("/{rubric_id}")
async def get_rubric(rubric_id: str, rubrics: RubricStoreDep) -> Rubric:
    if rubric_id not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rubric_id}' not found")
    return rubrics[rubric_id]


@router.post("", status_code=201)
async def create_rubric(rubric: Rubric, rubrics: RubricStoreDep) -> Rubric:
    rubrics[rubric.id] = rubric
    return rubric
