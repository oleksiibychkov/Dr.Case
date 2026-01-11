"""
Dr.Case — Схеми стану ітерації

Pydantic моделі для:
- SOMResult: результат проєкції на SOM
- Question: уточнююче питання
- IterationState: повний стан ітерації R^(t)
- SessionState: стан всієї сесії діагностики
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np


class AnswerType(str, Enum):
    """Тип відповіді на питання"""
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class SOMResult(BaseModel):
    """
    Результат проєкції пацієнта на SOM.
    
    Містить membership values для юнітів карти.
    """
    # BMU (Best Matching Unit)
    bmu_coords: tuple = Field(..., description="Координати BMU (row, col)")
    bmu_distance: float = Field(..., description="Відстань до BMU")
    
    # Membership для всіх активних юнітів
    memberships: Dict[str, float] = Field(
        default_factory=dict,
        description="Membership для юнітів {unit_id: value}"
    )
    
    # Топ юніти
    top_units: List[str] = Field(
        default_factory=list,
        description="Список топ юнітів за membership"
    )
    
    # Кількість юнітів з ненульовим membership
    active_units_count: int = Field(default=0)
    
    # Cumulative mass досягнутий
    cumulative_mass: float = Field(
        default=0.0,
        description="Сума membership топ юнітів"
    )
    
    @property
    def bmu_id(self) -> str:
        """ID BMU у форматі 'row_col'"""
        return f"{self.bmu_coords[0]}_{self.bmu_coords[1]}"
    
    def get_membership(self, unit_id: str) -> float:
        """Отримати membership для юніта"""
        return self.memberships.get(unit_id, 0.0)
    
    class Config:
        # Дозволяємо tuple
        arbitrary_types_allowed = True
        
        json_schema_extra = {
            "example": {
                "bmu_coords": [3, 4],
                "bmu_distance": 0.23,
                "memberships": {"3_4": 0.45, "3_5": 0.30, "4_4": 0.25},
                "top_units": ["3_4", "3_5", "4_4"],
                "active_units_count": 3,
                "cumulative_mass": 1.0
            }
        }


class Question(BaseModel):
    """
    Уточнююче питання для пацієнта.
    """
    symptom: str = Field(..., description="Симптом про який питаємо")
    text: str = Field(..., description="Текст питання для відображення")
    
    # Information Gain
    information_gain: float = Field(
        default=0.0,
        ge=0.0,
        description="Очікуваний Information Gain"
    )
    
    # Пояснення
    explanation: Optional[str] = Field(
        default=None,
        description="Чому це питання важливе"
    )
    
    # Відповідь (заповнюється після)
    answer: Optional[AnswerType] = Field(default=None)
    answered_at: Optional[datetime] = Field(default=None)
    
    def set_answer(self, answer: AnswerType) -> None:
        """Встановити відповідь"""
        self.answer = answer
        self.answered_at = datetime.now()
    
    @property
    def is_answered(self) -> bool:
        """Чи є відповідь"""
        return self.answer is not None
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptom": "muscle_pain",
                "text": "Чи відчуваєте ви біль у м'язах або ломоту в тілі?",
                "information_gain": 0.34,
                "explanation": "Біль у м'язах характерний для грипу (85%) і менш типовий для ГРВІ (30%)"
            }
        }


class IterationState(BaseModel):
    """
    Стан однієї ітерації діагностики R^(t).
    
    Відповідає формулі з проекту:
    R^(t) = (x^(t), m^(t), D_cand^(t), ŷ^(t))
    
    Де:
    - x^(t) — вектор симптомів пацієнта
    - m^(t) — membership values від SOM
    - D_cand^(t) — список діагнозів-кандидатів
    - ŷ^(t) — ранжування діагнозів від NN
    """
    # Номер ітерації
    iteration: int = Field(..., ge=0, description="Номер ітерації t")
    
    # Симптоми (x^(t))
    present_symptoms: List[str] = Field(
        default_factory=list,
        description="Присутні симптоми"
    )
    absent_symptoms: List[str] = Field(
        default_factory=list,
        description="Відсутні симптоми"
    )
    
    # Вектор симптомів (для передачі в numpy)
    # Зберігаємо як список для серіалізації
    symptom_vector: Optional[List[float]] = Field(
        default=None,
        description="Вектор симптомів x ∈ ℝ^D"
    )
    
    # SOM результат (m^(t))
    som_result: Optional[SOMResult] = Field(default=None)
    
    # Кандидати (D_cand^(t))
    candidate_diseases: List[str] = Field(
        default_factory=list,
        description="Діагнози-кандидати"
    )
    
    # Ранжування (ŷ^(t))
    disease_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores від NN {disease: score}"
    )
    
    # Питання цієї ітерації
    question: Optional[Question] = Field(default=None)
    
    # Час
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def symptom_count(self) -> int:
        """Кількість відомих симптомів"""
        return len(self.present_symptoms) + len(self.absent_symptoms)
    
    @property
    def candidate_count(self) -> int:
        """Кількість кандидатів"""
        return len(self.candidate_diseases)
    
    @property
    def top_disease(self) -> Optional[str]:
        """Топ діагноз за score"""
        if not self.disease_scores:
            return None
        return max(self.disease_scores.items(), key=lambda x: x[1])[0]
    
    @property
    def top_score(self) -> float:
        """Score топ діагнозу"""
        if not self.disease_scores:
            return 0.0
        return max(self.disease_scores.values())
    
    def get_top_diseases(self, n: int = 5) -> List[tuple]:
        """Отримати топ-N діагнозів"""
        sorted_diseases = sorted(
            self.disease_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_diseases[:n]
    
    def to_numpy_vector(self) -> Optional[np.ndarray]:
        """Конвертувати symptom_vector в numpy array"""
        if self.symptom_vector is None:
            return None
        return np.array(self.symptom_vector, dtype=np.float32)
    
    class Config:
        arbitrary_types_allowed = True
        
        json_schema_extra = {
            "example": {
                "iteration": 3,
                "present_symptoms": ["fever", "headache", "cough"],
                "absent_symptoms": ["rash"],
                "candidate_diseases": ["Influenza", "Common Cold", "COVID-19"],
                "disease_scores": {
                    "Influenza": 0.78,
                    "Common Cold": 0.45,
                    "COVID-19": 0.32
                }
            }
        }


class SessionState(BaseModel):
    """
    Повний стан сесії діагностики.
    
    Зберігає історію всіх ітерацій та поточний стан.
    """
    session_id: str = Field(..., description="ID сесії")
    case_id: Optional[str] = Field(default=None, description="ID випадку")
    
    # Поточна ітерація
    current_iteration: int = Field(default=0)
    
    # Історія ітерацій
    iterations: List[IterationState] = Field(
        default_factory=list,
        description="Історія всіх ітерацій"
    )
    
    # Статус
    is_active: bool = Field(default=True, description="Чи сесія активна")
    is_waiting_answer: bool = Field(default=False, description="Чи очікує відповіді")
    
    # Виключені діагнози
    excluded_diseases: List[str] = Field(
        default_factory=list,
        description="Виключені діагнози"
    )
    exclusion_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Причини виключення"
    )
    
    # Час
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default=None)
    
    @property
    def latest_iteration(self) -> Optional[IterationState]:
        """Остання ітерація"""
        return self.iterations[-1] if self.iterations else None
    
    @property
    def questions_asked(self) -> int:
        """Кількість заданих питань"""
        return sum(1 for it in self.iterations if it.question and it.question.is_answered)
    
    @property
    def duration_seconds(self) -> float:
        """Тривалість сесії в секундах"""
        end_time = self.updated_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def add_iteration(self, iteration: IterationState) -> None:
        """Додати ітерацію"""
        self.iterations.append(iteration)
        self.current_iteration = iteration.iteration
        self.updated_at = datetime.now()
    
    def exclude_disease(self, disease: str, reason: str) -> None:
        """Виключити діагноз"""
        if disease not in self.excluded_diseases:
            self.excluded_diseases.append(disease)
        self.exclusion_reasons[disease] = reason
        self.updated_at = datetime.now()
    
    def restore_excluded(self) -> List[str]:
        """Повернути виключені діагнози (при невдачі лікування)"""
        restored = self.excluded_diseases.copy()
        self.excluded_diseases.clear()
        self.exclusion_reasons.clear()
        self.updated_at = datetime.now()
        return restored
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_12345",
                "case_id": "case_001",
                "current_iteration": 3,
                "is_active": True,
                "questions_asked": 4
            }
        }
