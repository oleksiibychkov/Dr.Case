"""
Dr.Case — API Models

Pydantic моделі для запитів та відповідей API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


# ============================================================
# Enums
# ============================================================

class Language(str, Enum):
    """Підтримувані мови"""
    UKRAINIAN = "uk"
    ENGLISH = "en"


class SessionStatus(str, Enum):
    """Статус сесії"""
    ACTIVE = "active"
    WAITING_ANSWER = "waiting_answer"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class StopReason(str, Enum):
    """Причина зупинки діагностики"""
    DOMINANCE = "dominance"
    STABILITY = "stability"
    NEED_TEST = "need_test"
    SAFETY_LIMIT = "safety_limit"
    USER_STOPPED = "user_stopped"


class FeedbackType(str, Enum):
    """Тип зворотного зв'язку"""
    TREATMENT_SUCCESS = "treatment_success"
    TREATMENT_FAILED = "treatment_failed"
    NEW_SYMPTOM = "new_symptom"
    CONDITION_CHANGED = "condition_changed"


# ============================================================
# Symptom Models
# ============================================================

class SymptomInfo(BaseModel):
    """Інформація про симптом"""
    name: str
    frequency: Optional[int] = None  # Скільки хвороб мають цей симптом


class SymptomSearchResponse(BaseModel):
    """Відповідь на пошук симптомів"""
    query: str
    results: List[SymptomInfo]
    total: int


class ExtractSymptomsRequest(BaseModel):
    """Запит на витягування симптомів з тексту"""
    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[Language] = None  # Auto-detect якщо None


class ExtractSymptomsResponse(BaseModel):
    """Відповідь з витягненими симптомами"""
    original_text: str
    symptoms: List[str]
    negated_symptoms: List[str] = []
    vitals: Dict[str, Any] = {}
    duration: Dict[str, int] = {}
    language: str
    confidence: float


# ============================================================
# Diagnosis Models
# ============================================================

class Hypothesis(BaseModel):
    """Гіпотеза діагнозу"""
    disease: str
    probability: float = Field(..., ge=0, le=1)
    rank: int
    change: Optional[float] = None  # Зміна з попередньої ітерації


class QuickDiagnoseRequest(BaseModel):
    """Запит на швидку діагностику"""
    symptoms: List[str] = Field(..., min_items=1)
    top_k: int = Field(default=10, ge=1, le=50)


class QuickDiagnoseResponse(BaseModel):
    """Відповідь швидкої діагностики"""
    symptoms: List[str]
    hypotheses: List[Hypothesis]
    processing_time_ms: float


# ============================================================
# Session Models
# ============================================================

class CreateSessionRequest(BaseModel):
    """Запит на створення сесії"""
    # Початкові симптоми (один з варіантів)
    symptoms: Optional[List[str]] = None
    text: Optional[str] = None  # Буде оброблено NLP
    
    # Опціональні дані
    patient_age: Optional[int] = Field(default=None, ge=0, le=150)
    patient_sex: Optional[str] = None
    language: Language = Language.UKRAINIAN


class SessionQuestion(BaseModel):
    """Питання для пацієнта"""
    symptom: str
    text_uk: str
    text_en: str
    explanation: Optional[str] = None


class SessionState(BaseModel):
    """Поточний стан сесії"""
    session_id: str
    status: SessionStatus
    iteration: int
    
    # Симптоми
    confirmed_symptoms: List[str]
    denied_symptoms: List[str]
    
    # Гіпотези
    hypotheses: List[Hypothesis]
    
    # Питання (якщо є)
    current_question: Optional[SessionQuestion] = None
    
    # Результат (якщо завершено)
    stop_reason: Optional[StopReason] = None
    final_diagnosis: Optional[str] = None
    
    # Метадані
    created_at: datetime
    updated_at: datetime


class AnswerRequest(BaseModel):
    """Відповідь на питання"""
    answer: Optional[bool] = None  # True=так, False=ні, None=не знаю


class AnswerResponse(BaseModel):
    """Результат обробки відповіді"""
    accepted: bool
    session_state: SessionState


class FeedbackRequest(BaseModel):
    """Зворотний зв'язок"""
    feedback_type: FeedbackType
    diagnosis: Optional[str] = None  # Для treatment feedback
    new_symptom: Optional[str] = None  # Для new_symptom
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Результат обробки feedback"""
    accepted: bool
    action_taken: str
    session_state: Optional[SessionState] = None


# ============================================================
# Health & Info Models
# ============================================================

class HealthResponse(BaseModel):
    """Відповідь health check"""
    status: str = "ok"
    version: str
    models_loaded: bool
    database_symptoms: int
    database_diseases: int
    active_sessions: int


class ErrorResponse(BaseModel):
    """Відповідь з помилкою"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
