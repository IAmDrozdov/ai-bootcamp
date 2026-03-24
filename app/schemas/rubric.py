from pydantic import BaseModel


class Criterion(BaseModel):
    name: str
    description: str
    max_score: int
    weight: float


class Rubric(BaseModel):
    id: str
    name: str
    criteria: list[Criterion]
