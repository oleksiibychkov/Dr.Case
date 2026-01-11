"""
Dr.Case — Сесія діагностики

DiagnosisSession зберігає стан діагностичного процесу:
- Історію симптомів (present/absent)
- Історію питань та відповідей
- Поточних кандидатів та їх scores
- Цикли діагностики
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from dr_case.schemas import (
    CaseRecord, Patient, Symptom, SymptomStatus,
    DiagnosisHypothesis, Question
)


class SessionStatus(Enum):
    """Статус сесії"""
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ConfidenceLevel(Enum):
    """Рівень впевненості в діагнозі"""
    LOW = "low"           # < 50%
    MEDIUM = "medium"     # 50-75%
    HIGH = "high"         # 75-90%
    VERY_HIGH = "very_high"  # > 90%
    
    @classmethod
    def from_confidence(cls, confidence: float) -> "ConfidenceLevel":
        if confidence < 0.5:
            return cls.LOW
        elif confidence < 0.75:
            return cls.MEDIUM
        elif confidence < 0.9:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


@dataclass
class QuestionAnswer:
    """Запис питання-відповідь"""
    question: Question
    answer: bool  # True = yes, False = no
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CycleResult:
    """Результат одного циклу діагностики"""
    cycle_number: int
    
    # Кандидати на початку циклу
    candidates_before: List[str]
    
    # Топ гіпотези після циклу
    top_hypotheses: List[DiagnosisHypothesis]
    
    # Питання та відповіді в цьому циклі
    questions_asked: List[QuestionAnswer]
    
    # Метрики
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    information_gained: float = 0.0


@dataclass
class DiagnosisSession:
    """
    Сесія діагностики.
    
    Зберігає весь стан діагностичного процесу.
    
    Приклад:
        session = DiagnosisSession()
        
        # Додаємо симптоми
        session.add_symptom("fever", present=True)
        session.add_symptom("rash", present=False)
        
        # Отримуємо поточний стан
        print(session.present_symptoms)
        print(session.absent_symptoms)
        
        # Записуємо відповідь на питання
        session.record_answer(question, answer=True)
    """
    
    # Ідентифікатор
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Інформація про пацієнта
    patient: Optional[Patient] = None
    chief_complaint: str = ""
    
    # Симптоми
    present_symptoms: List[str] = field(default_factory=list)
    absent_symptoms: List[str] = field(default_factory=list)
    
    # Історія питань
    question_history: List[QuestionAnswer] = field(default_factory=list)
    
    # Поточні кандидати та scores
    current_candidates: List[str] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    
    # Топ гіпотези
    top_hypotheses: List[DiagnosisHypothesis] = field(default_factory=list)
    
    # Цикли діагностики
    cycle_history: List[CycleResult] = field(default_factory=list)
    current_cycle: int = 0
    
    # Статус
    status: SessionStatus = SessionStatus.ACTIVE
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Налаштування
    max_cycles: int = 5
    max_questions_per_cycle: int = 3
    confidence_threshold: float = 0.8
    
    def add_symptom(self, symptom: str, present: bool = True) -> None:
        """
        Додати симптом.
        
        Args:
            symptom: Назва симптому
            present: True = присутній, False = відсутній
        """
        # Видаляємо з обох списків (якщо був)
        if symptom in self.present_symptoms:
            self.present_symptoms.remove(symptom)
        if symptom in self.absent_symptoms:
            self.absent_symptoms.remove(symptom)
        
        # Додаємо в правильний список
        if present:
            self.present_symptoms.append(symptom)
        else:
            self.absent_symptoms.append(symptom)
        
        self.updated_at = datetime.now()
    
    def add_symptoms(self, symptoms: List[str], present: bool = True) -> None:
        """Додати кілька симптомів"""
        for symptom in symptoms:
            self.add_symptom(symptom, present)
    
    def record_answer(self, question: Question, answer: bool) -> None:
        """
        Записати відповідь на питання.
        
        Args:
            question: Питання
            answer: True = так, False = ні
        """
        qa = QuestionAnswer(question=question, answer=answer)
        self.question_history.append(qa)
        
        # Додаємо симптом
        self.add_symptom(question.symptom, present=answer)
        
        self.updated_at = datetime.now()
    
    def update_candidates(
        self,
        candidates: List[str],
        scores: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Оновити список кандидатів.
        
        Args:
            candidates: Нові кандидати
            scores: Scores кандидатів
        """
        self.current_candidates = candidates.copy()
        self.candidate_scores = scores.copy() if scores else {}
        self.updated_at = datetime.now()
    
    def update_hypotheses(self, hypotheses: List[DiagnosisHypothesis]) -> None:
        """Оновити топ гіпотези"""
        self.top_hypotheses = hypotheses.copy()
        self.updated_at = datetime.now()
    
    def start_new_cycle(self) -> int:
        """Почати новий цикл діагностики"""
        self.current_cycle += 1
        return self.current_cycle
    
    def complete_cycle(self, cycle_result: CycleResult) -> None:
        """Завершити цикл"""
        self.cycle_history.append(cycle_result)
        self.updated_at = datetime.now()
    
    def complete(self) -> None:
        """Завершити сесію"""
        self.status = SessionStatus.COMPLETED
        self.updated_at = datetime.now()
    
    def cancel(self) -> None:
        """Скасувати сесію"""
        self.status = SessionStatus.CANCELLED
        self.updated_at = datetime.now()
    
    @property
    def all_known_symptoms(self) -> List[str]:
        """Всі відомі симптоми (present + absent)"""
        return self.present_symptoms + self.absent_symptoms
    
    @property
    def n_questions_asked(self) -> int:
        """Кількість заданих питань"""
        return len(self.question_history)
    
    @property
    def n_candidates(self) -> int:
        """Кількість поточних кандидатів"""
        return len(self.current_candidates)
    
    @property
    def top_diagnosis(self) -> Optional[DiagnosisHypothesis]:
        """Топ діагноз"""
        return self.top_hypotheses[0] if self.top_hypotheses else None
    
    @property
    def top_confidence(self) -> float:
        """Впевненість в топ діагнозі"""
        return self.top_diagnosis.confidence if self.top_diagnosis else 0.0
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Рівень впевненості"""
        return ConfidenceLevel.from_confidence(self.top_confidence)
    
    @property
    def should_continue(self) -> bool:
        """Чи потрібно продовжувати діагностику"""
        if self.status != SessionStatus.ACTIVE:
            return False
        if self.current_cycle >= self.max_cycles:
            return False
        if self.top_confidence >= self.confidence_threshold:
            return False
        if self.n_candidates <= 1:
            return False
        return True
    
    @property
    def is_active(self) -> bool:
        """Чи сесія активна"""
        return self.status == SessionStatus.ACTIVE
    
    def to_case_record(self) -> CaseRecord:
        """Конвертувати в CaseRecord"""
        symptoms = []
        
        for s in self.present_symptoms:
            symptoms.append(Symptom(name=s, status=SymptomStatus.PRESENT))
        
        for s in self.absent_symptoms:
            symptoms.append(Symptom(name=s, status=SymptomStatus.ABSENT))
        
        return CaseRecord(
            case_id=self.session_id,
            patient=self.patient,
            chief_complaint=self.chief_complaint,
            symptoms=symptoms
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Отримати підсумок сесії"""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "cycles": self.current_cycle,
            "questions_asked": self.n_questions_asked,
            "present_symptoms": self.present_symptoms,
            "absent_symptoms": self.absent_symptoms,
            "candidates": self.n_candidates,
            "top_diagnosis": self.top_diagnosis.disease_name if self.top_diagnosis else None,
            "top_confidence": self.top_confidence,
            "confidence_level": self.confidence_level.value,
            "duration_seconds": (self.updated_at - self.created_at).total_seconds(),
        }
    
    def __repr__(self) -> str:
        return (
            f"DiagnosisSession("
            f"id={self.session_id}, "
            f"cycle={self.current_cycle}, "
            f"symptoms={len(self.present_symptoms)}+/{len(self.absent_symptoms)}-, "
            f"candidates={self.n_candidates}, "
            f"top={self.top_diagnosis.disease_name if self.top_diagnosis else 'None'}"
            f")"
        )
