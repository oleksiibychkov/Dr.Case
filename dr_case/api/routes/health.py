"""
Dr.Case — Health Routes

Health check та інформація про систему.
"""

from fastapi import APIRouter, Depends

from ..dependencies import get_models, get_sessions, ModelsManager, SessionManager
from ..models import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    models: ModelsManager = Depends(get_models),
    sessions: SessionManager = Depends(get_sessions)
) -> HealthResponse:
    """
    Перевірка стану сервера.
    
    Повертає:
    - Статус сервера
    - Чи завантажені моделі
    - Кількість симптомів/хвороб в базі
    - Кількість активних сесій
    """
    return HealthResponse(
        status="ok" if models.is_loaded else "degraded",
        version="1.0.0",
        models_loaded=models.is_loaded,
        database_symptoms=len(models.symptom_list),
        database_diseases=len(models.disease_list),
        active_sessions=sessions.get_active_count()
    )


@router.get("/")
async def root():
    """Головна сторінка API"""
    return {
        "name": "Dr.Case API",
        "version": "1.0.0",
        "description": "Інтелектуальна система медичної діагностики",
        "docs": "/docs",
        "health": "/health",
    }
