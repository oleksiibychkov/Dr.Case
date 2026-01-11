"""
Dr.Case — REST API модуль

FastAPI REST API для діагностичної системи.

Компоненти:
- main.py: FastAPI application
- routes.py: API endpoints
- models.py: Pydantic models
- dependencies.py: Залежності та стан

Запуск:
    cd C:\\Projects\\Dr.Case
    uvicorn dr_case.api.main:app --reload --port 8000

Або:
    python -m dr_case.api.main

Документація:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Endpoints:
    GET  /                          - Root info
    GET  /api/v1/health            - Health check
    
    POST /api/v1/diagnose/quick    - Швидка діагностика
    
    POST /api/v1/session/start     - Почати сесію
    GET  /api/v1/session/{id}      - Інфо про сесію
    GET  /api/v1/session/{id}/questions - Отримати питання
    POST /api/v1/session/{id}/answer    - Відповісти на питання
    POST /api/v1/session/{id}/complete  - Завершити сесію
    
    GET  /api/v1/info/symptoms     - Список симптомів
    GET  /api/v1/info/diseases     - Список захворювань
    GET  /api/v1/info/symptoms/search?q= - Пошук симптомів
    GET  /api/v1/info/diseases/search?q= - Пошук захворювань
"""

from .main import app
from .dependencies import app_state, get_engine, get_session_store


__all__ = [
    "app",
    "app_state",
    "get_engine",
    "get_session_store",
]
