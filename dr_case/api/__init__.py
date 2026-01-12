"""
Dr.Case — API Module

FastAPI backend для системи медичної діагностики.

Запуск:
    python scripts/run_api.py
    
    або:
    
    uvicorn dr_case.api.app:app --reload

Документація:
    http://localhost:8000/docs

Endpoints:
    GET  /health              - Перевірка стану
    GET  /api/symptoms        - Список симптомів
    GET  /api/symptoms/search - Пошук симптомів
    POST /api/symptoms/extract - NLP витягування симптомів
    POST /api/diagnose        - Швидка діагностика
    POST /api/diagnose/text   - Діагностика з тексту
    POST /api/sessions        - Створити сесію
    GET  /api/sessions/{id}   - Стан сесії
    POST /api/sessions/{id}/answer - Відповідь на питання
    POST /api/sessions/{id}/feedback - Зворотний зв'язок
    DELETE /api/sessions/{id} - Видалити сесію
"""

from .app import app
from .config import config, APIConfig
from .dependencies import (
    ModelsManager,
    SessionManager,
    DiagnosisSession,
    models_manager,
    session_manager,
    get_models,
    get_sessions,
)

__all__ = [
    'app',
    'config',
    'APIConfig',
    'ModelsManager',
    'SessionManager',
    'DiagnosisSession',
    'models_manager',
    'session_manager',
    'get_models',
    'get_sessions',
]
