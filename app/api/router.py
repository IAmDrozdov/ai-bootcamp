from fastapi import APIRouter

from app.api.v1 import assessment, rubrics

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(assessment.router)
api_router.include_router(rubrics.router)
