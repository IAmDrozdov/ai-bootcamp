from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.dependencies import ChainDep, LLMDep, RubricStoreDep
from app.schemas.assessment import AssessmentRequest, AssessmentResponse
from app.services.assessment import assess, assess_stream

router = APIRouter(prefix="/assess", tags=["assessment"])


@router.post("")
async def assess_work(
    request: AssessmentRequest,
    chain: ChainDep,
    rubrics: RubricStoreDep,
) -> AssessmentResponse:
    rubric = _resolve_rubric(request, rubrics)
    return await assess(request.student_work, rubric, chain)


@router.post("/stream")
async def assess_work_stream(
    request: AssessmentRequest,
    llm: LLMDep,
    rubrics: RubricStoreDep,
) -> EventSourceResponse:
    rubric = _resolve_rubric(request, rubrics)

    async def event_generator():
        async for token in assess_stream(request.student_work, rubric, llm):
            yield {"data": token}

    return EventSourceResponse(event_generator())


def _resolve_rubric(request, rubrics):
    if request.rubric:
        return request.rubric
    rubric_id = request.rubric_id or "essay_default"
    if rubric_id not in rubrics:
        raise HTTPException(status_code=404, detail=f"Rubric '{rubric_id}' not found")
    return rubrics[rubric_id]
