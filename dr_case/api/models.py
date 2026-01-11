"""
Dr.Case API — Pydantic Models

Моделі для запитів та відповідей REST API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# === Enums ===

class ConfidenceLevelEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SessionStatusEnum(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# === Request Models ===

class QuickDiagnoseRequest(BaseModel):
    """Запит на швидку діагностику"""
    present_symptoms: List[str] = Field(
        ...,
        description="Список присутніх симптомів",
        example=["fever", "cough", "headache"]
    )
    absent_symptoms: Optional[List[str]] = Field(
        default=None,
        description="Список відсутніх симптомів",
        example=["rash", "vomiting"]
    )


class StartSessionRequest(BaseModel):
    """Запит на початок сесії"""
    initial_symptoms: List[str] = Field(
        ...,
        description="Початкові симптоми",
        example=["fever", "cough"]
    )
    patient_age: Optional[int] = Field(
        default=None,
        description="Вік пацієнта"
    )
    patient_gender: Optional[str] = Field(
        default=None,
        description="Стать пацієнта (male/female)"
    )
    chief_complaint: Optional[str] = Field(
        default=None,
        description="Головна скарга пацієнта"
    )


class AnswerRequest(BaseModel):
    """Запит з відповіддю на питання"""
    symptom: str = Field(
        ...,
        description="Симптом з питання"
    )
    answer: bool = Field(
        ...,
        description="Відповідь: true = так, false = ні"
    )


class BatchAnswerRequest(BaseModel):
    """Запит з кількома відповідями"""
    answers: Dict[str, bool] = Field(
        ...,
        description="Словник {symptom: answer}",
        example={"headache": True, "rash": False}
    )


# === Response Models ===

class DiagnosisHypothesisResponse(BaseModel):
    """Гіпотеза діагнозу"""
    disease_name: str = Field(..., description="Назва захворювання")
    confidence: float = Field(..., description="Впевненість (0-1)")
    confidence_percent: str = Field(..., description="Впевненість у відсотках")
    matching_symptoms: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease_name": "Influenza",
                "confidence": 0.45,
                "confidence_percent": "45.0%",
                "matching_symptoms": ["fever", "cough", "headache"]
            }
        }


class QuestionResponse(BaseModel):
    """Питання для користувача"""
    symptom: str = Field(..., description="Симптом")
    text: str = Field(..., description="Текст питання")
    information_gain: float = Field(..., description="Information Gain")
    explanation: Optional[str] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptom": "chest_pain",
                "text": "Do you have chest pain?",
                "information_gain": 0.95,
                "explanation": "This symptom helps distinguish between 5 and 3 diagnoses."
            }
        }


class QuickDiagnoseResponse(BaseModel):
    """Результат швидкої діагностики"""
    hypotheses: List[DiagnosisHypothesisResponse]
    present_symptoms: List[str]
    absent_symptoms: List[str]
    candidates_count: int
    top_diagnosis: Optional[str]
    top_confidence: float
    confidence_level: ConfidenceLevelEnum
    
    class Config:
        json_schema_extra = {
            "example": {
                "hypotheses": [
                    {"disease_name": "Influenza", "confidence": 0.45, "confidence_percent": "45.0%", "matching_symptoms": ["fever", "cough"]},
                    {"disease_name": "Common Cold", "confidence": 0.30, "confidence_percent": "30.0%", "matching_symptoms": ["fever", "cough"]}
                ],
                "present_symptoms": ["fever", "cough", "headache"],
                "absent_symptoms": [],
                "candidates_count": 13,
                "top_diagnosis": "Influenza",
                "top_confidence": 0.45,
                "confidence_level": "medium"
            }
        }


class SessionResponse(BaseModel):
    """Інформація про сесію"""
    session_id: str
    status: SessionStatusEnum
    current_cycle: int
    questions_asked: int
    present_symptoms: List[str]
    absent_symptoms: List[str]
    candidates_count: int
    top_diagnosis: Optional[str]
    top_confidence: float
    confidence_level: ConfidenceLevelEnum
    should_continue: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "status": "active",
                "current_cycle": 1,
                "questions_asked": 3,
                "present_symptoms": ["fever", "cough"],
                "absent_symptoms": ["rash"],
                "candidates_count": 8,
                "top_diagnosis": "Influenza",
                "top_confidence": 0.65,
                "confidence_level": "medium",
                "should_continue": True
            }
        }


class SessionWithQuestionsResponse(SessionResponse):
    """Сесія з питаннями"""
    next_questions: List[QuestionResponse] = Field(default_factory=list)


class DiagnosisResultResponse(BaseModel):
    """Фінальний результат діагностики"""
    session_id: str
    status: SessionStatusEnum
    cycles_completed: int
    questions_asked: int
    
    hypotheses: List[DiagnosisHypothesisResponse]
    present_symptoms: List[str]
    absent_symptoms: List[str]
    
    top_diagnosis: Optional[str]
    top_confidence: float
    confidence_level: ConfidenceLevelEnum
    is_confident: bool
    
    explanation: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    version: str = "1.0.0"
    components: Dict[str, bool] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "components": {
                    "som_model": True,
                    "nn_model": True,
                    "database": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Помилка"""
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Session not found",
                "detail": "Session with id 'xyz' does not exist"
            }
        }


class SymptomListResponse(BaseModel):
    """Список симптомів"""
    symptoms: List[str]
    count: int


class DiseaseListResponse(BaseModel):
    """Список захворювань"""
    diseases: List[str]
    count: int
