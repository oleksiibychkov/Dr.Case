"""
Dr.Case — API Routes

Експорт всіх роутерів.
"""

from .health import router as health_router
from .symptoms import router as symptoms_router
from .diagnosis import router as diagnosis_router
from .sessions import router as sessions_router

__all__ = [
    'health_router',
    'symptoms_router',
    'diagnosis_router',
    'sessions_router',
]
