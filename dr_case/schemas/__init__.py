"""
Dr.Case — Модуль схем даних (schemas)

Pydantic моделі для валідації та серіалізації даних.

Компоненти:
- patient.py: Symptom, Patient, Vitals, CaseRecord
- diagnosis.py: DiagnosisHypothesis, CandidateDiagnoses, DiagnosisResult
- iteration.py: SOMResult, Question, IterationState, SessionState

Приклад використання:
    from dr_case.schemas import (
        CaseRecord, Patient, Symptom, SymptomStatus, Gender,
        DiagnosisResult, DiagnosisHypothesis,
        IterationState, SessionState
    )
    
    # Створення випадку пацієнта
    case = CaseRecord(
        patient=Patient(age=35, gender=Gender.MALE),
        chief_complaint="Головний біль та температура",
        symptoms=[
            Symptom(name="fever", severity=0.8),
            Symptom(name="headache", severity=0.7),
        ]
    )
    
    # Серіалізація в JSON
    json_data = case.model_dump_json()
    
    # Десеріалізація з JSON
    case_loaded = CaseRecord.model_validate_json(json_data)
"""

# Patient schemas
from .patient import (
    Gender,
    SymptomStatus,
    Symptom,
    Vitals,
    Patient,
    CaseRecord,
)

# Diagnosis schemas
from .diagnosis import (
    StopReason,
    ConfidenceLevel,
    DiagnosisHypothesis,
    CandidateDiagnoses,
    DifferentialEntry,
    RecommendedTest,
    DiagnosisResult,
)

# Iteration schemas
from .iteration import (
    AnswerType,
    SOMResult,
    Question,
    IterationState,
    SessionState,
)


__all__ = [
    # Patient
    "Gender",
    "SymptomStatus",
    "Symptom",
    "Vitals",
    "Patient",
    "CaseRecord",
    
    # Diagnosis
    "StopReason",
    "ConfidenceLevel",
    "DiagnosisHypothesis",
    "CandidateDiagnoses",
    "DifferentialEntry",
    "RecommendedTest",
    "DiagnosisResult",
    
    # Iteration
    "AnswerType",
    "SOMResult",
    "Question",
    "IterationState",
    "SessionState",
]
