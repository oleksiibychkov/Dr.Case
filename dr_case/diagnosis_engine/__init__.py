"""
Dr.Case — Головний діагностичний движок (Diagnosis Engine)

Об'єднує всі компоненти системи:
- CandidateSelector (SOM) — звуження простору пошуку
- DiagnosisRanker (NN) — ранжування кандидатів  
- QuestionGenerator — генерація уточнюючих питань

Компоненти:
- DiagnosisEngine: Головний клас для діагностики
- DiagnosisSession: Сесія діагностичного процесу
- DiagnosisResult: Результат діагностики

Приклад використання:
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    # Завантаження
    engine = DiagnosisEngine.from_models(
        som_model_path="models/som_optimized.pkl",
        database_path="data/unified_disease_symptom_data_full.json",
        nn_model_path="models/nn_model.pt"  # опціонально
    )
    
    # === Швидка діагностика ===
    result = engine.diagnose_quick(["fever", "cough", "headache"])
    
    print(f"Top diagnosis: {result.top_diagnosis.disease_name}")
    print(f"Confidence: {result.top_confidence:.1%}")
    
    for h in result.get_top_n(5):
        print(f"  {h.disease_name}: {h.confidence:.1%}")
    
    # === Інтерактивна діагностика ===
    session = engine.start_session(["fever", "cough"])
    
    while session.should_continue:
        question = engine.get_next_question(session)
        print(f"Q: {question.text}")
        
        answer = input("Answer (y/n): ").lower() == 'y'
        engine.process_answer(session, question, answer)
        
        print(f"Top: {session.top_diagnosis.disease_name} ({session.top_confidence:.1%})")
    
    result = engine.get_result(session)
    print(engine.explain_diagnosis(session))
"""

from .session import (
    DiagnosisSession,
    SessionStatus,
    ConfidenceLevel,
    CycleResult,
    QuestionAnswer
)
from .engine import (
    DiagnosisEngine,
    DiagnosisResult
)


__all__ = [
    # Engine
    "DiagnosisEngine",
    "DiagnosisResult",
    
    # Session
    "DiagnosisSession",
    "SessionStatus",
    "ConfidenceLevel",
    "CycleResult",
    "QuestionAnswer",
]
