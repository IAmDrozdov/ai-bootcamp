from pydantic import BaseModel, Field

from app.schemas.rubric import Rubric


class CriterionScore(BaseModel):
    criterion_name: str
    score: int
    max_score: int
    feedback: str


class AssessmentResponse(BaseModel):
    """Structured output schema for LLM assessment.

    LangChain's with_structured_output() uses this Pydantic model to constrain
    the LLM response into a predictable JSON shape — key pattern for integrating
    LLM into applications where downstream code needs typed data, not free text.
    """

    overall_score: int = Field(description="Total score across all criteria")
    max_overall_score: int = Field(description="Maximum possible total score")
    criterion_scores: list[CriterionScore] = Field(description="Per-criterion breakdown")
    summary: str = Field(description="Brief overall assessment summary")
    strengths: list[str] = Field(description="Key strengths of the work")
    improvements: list[str] = Field(description="Suggested areas for improvement")


class AssessmentRequest(BaseModel):
    student_work: str
    rubric_id: str | None = "essay_default"
    rubric: Rubric | None = None
