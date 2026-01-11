"""
Dr.Case — Схеми результатів діагностики

Pydantic моделі для:
- DiagnosisHypothesis: одна гіпотеза діагнозу
- CandidateDiagnoses: список кандидатів від Candidate Selector
- DiagnosisResult: фінальний результат діагностики
- DifferentialDiagnosis: диференціальний діагноз
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class StopReason(str, Enum):
    """Причина зупинки діагностики"""
    DOMINANCE = "dominance"           # є чіткий лідер
    STABILITY = "stability"           # гіпотези стабілізувались
    NEED_TEST = "need_test"           # потрібні аналізи
    SAFETY_LIMIT = "safety_limit"     # досягнуто ліміту ітерацій
    USER_STOP = "user_stop"           # користувач зупинив
    ESCALATION = "escalation"         # ескалація до лікаря


class ConfidenceLevel(str, Enum):
    """Рівень впевненості"""
    HIGH = "high"           # > 0.8
    MEDIUM = "medium"       # 0.5 - 0.8
    LOW = "low"             # 0.3 - 0.5
    VERY_LOW = "very_low"   # < 0.3


class DiagnosisHypothesis(BaseModel):
    """
    Одна гіпотеза діагнозу.
    
    Приклад:
        hypothesis = DiagnosisHypothesis(
            disease_name="Influenza",
            confidence=0.85,
            matching_symptoms=["fever", "headache", "cough"]
        )
    """
    disease_name: str = Field(..., description="Назва діагнозу")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Впевненість [0, 1]"
    )
    
    # Симптоми
    matching_symptoms: List[str] = Field(
        default_factory=list,
        description="Симптоми що співпали"
    )
    missing_symptoms: List[str] = Field(
        default_factory=list,
        description="Типові симптоми що відсутні"
    )
    extra_symptoms: List[str] = Field(
        default_factory=list,
        description="Симптоми пацієнта не характерні для діагнозу"
    )
    
    # Зміна від попередньої ітерації
    confidence_change: float = Field(
        default=0.0,
        description="Зміна впевненості від попередньої ітерації"
    )
    
    # Пояснення
    explanation: Optional[str] = Field(
        default=None,
        description="Пояснення чому цей діагноз"
    )
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Категорія впевненості"""
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence > 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    @property
    def trend(self) -> str:
        """Тренд: зростає/падає/стабільний"""
        if self.confidence_change > 0.05:
            return "rising"
        elif self.confidence_change < -0.05:
            return "falling"
        else:
            return "stable"
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease_name": "Influenza",
                "confidence": 0.85,
                "matching_symptoms": ["fever", "headache", "cough", "fatigue"],
                "missing_symptoms": ["runny nose"],
                "confidence_change": 0.12
            }
        }


class CandidateDiagnoses(BaseModel):
    """
    Результат роботи Candidate Selector.
    
    Містить список діагнозів-кандидатів відібраних SOM.
    """
    candidates: List[str] = Field(
        default_factory=list,
        description="Список назв діагнозів-кандидатів"
    )
    
    # Membership значення для кожного юніта SOM
    unit_memberships: Dict[str, float] = Field(
        default_factory=dict,
        description="Membership для кожного юніта {unit_id: membership}"
    )
    
    # Параметри відбору
    alpha: float = Field(default=0.9, description="Параметр cumulative mass")
    k: int = Field(default=6, description="Max кількість юнітів")
    tau: float = Field(default=0.01, description="Мінімальний поріг membership")
    
    # BMU (Best Matching Unit)
    bmu_id: Optional[str] = Field(default=None, description="ID найближчого юніта")
    bmu_distance: Optional[float] = Field(default=None, description="Відстань до BMU")
    
    @property
    def candidate_count(self) -> int:
        """Кількість кандидатів"""
        return len(self.candidates)
    
    @property
    def active_units(self) -> int:
        """Кількість активних юнітів"""
        return len(self.unit_memberships)
    
    def contains(self, disease_name: str) -> bool:
        """Перевірити чи діагноз є серед кандидатів"""
        return disease_name in self.candidates
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidates": ["Influenza", "Common Cold", "COVID-19", "Bronchitis"],
                "unit_memberships": {"3_4": 0.45, "3_5": 0.30, "4_4": 0.25},
                "alpha": 0.9,
                "k": 6,
                "bmu_id": "3_4",
                "bmu_distance": 0.23
            }
        }


class DifferentialEntry(BaseModel):
    """Запис у диференціальному діагнозі"""
    disease_name: str
    confidence: float
    difference_from_top: str = Field(
        default="",
        description="Чим відрізняється від основного діагнозу"
    )


class RecommendedTest(BaseModel):
    """Рекомендований аналіз/обстеження"""
    name: str = Field(..., description="Назва аналізу")
    reason: str = Field(default="", description="Причина призначення")
    priority: str = Field(default="routine", description="Пріоритет: urgent/routine")
    differentiates: List[str] = Field(
        default_factory=list,
        description="Які діагнози допоможе розрізнити"
    )


class DiagnosisResult(BaseModel):
    """
    Фінальний результат діагностики.
    
    Повертається після завершення циклу діагностики.
    """
    # Ідентифікатори
    case_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    
    # Статус
    stop_reason: StopReason = Field(..., description="Причина зупинки")
    is_final: bool = Field(default=True, description="Чи це фінальний результат")
    
    # Основний діагноз
    top_diagnosis: Optional[DiagnosisHypothesis] = Field(
        default=None,
        description="Найбільш ймовірний діагноз"
    )
    
    # Всі гіпотези (відсортовані за confidence)
    hypotheses: List[DiagnosisHypothesis] = Field(
        default_factory=list,
        description="Всі гіпотези"
    )
    
    # Диференціальний діагноз
    differential: List[DifferentialEntry] = Field(
        default_factory=list,
        description="Диференціальний діагноз"
    )
    
    # Рекомендації
    recommended_tests: List[RecommendedTest] = Field(
        default_factory=list,
        description="Рекомендовані обстеження"
    )
    
    # Статистика сесії
    iterations_count: int = Field(default=0, description="Кількість ітерацій")
    questions_asked: int = Field(default=0, description="Кількість питань")
    duration_seconds: float = Field(default=0.0, description="Тривалість (сек)")
    
    # Метадані
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def top_confidence(self) -> float:
        """Впевненість у топ-діагнозі"""
        return self.top_diagnosis.confidence if self.top_diagnosis else 0.0
    
    @property
    def is_confident(self) -> bool:
        """Чи є висока впевненість"""
        return self.top_confidence > 0.8
    
    def get_top_n(self, n: int = 5) -> List[DiagnosisHypothesis]:
        """Отримати топ-N гіпотез"""
        return sorted(self.hypotheses, key=lambda x: x.confidence, reverse=True)[:n]
    
    def to_summary(self) -> Dict:
        """Короткий підсумок для UI"""
        return {
            "top_diagnosis": self.top_diagnosis.disease_name if self.top_diagnosis else None,
            "confidence": self.top_confidence,
            "stop_reason": self.stop_reason.value,
            "iterations": self.iterations_count,
            "differential_count": len(self.differential),
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "stop_reason": "dominance",
                "top_diagnosis": {
                    "disease_name": "Influenza",
                    "confidence": 0.87,
                    "matching_symptoms": ["fever", "headache", "cough"]
                },
                "iterations_count": 5,
                "questions_asked": 4
            }
        }
