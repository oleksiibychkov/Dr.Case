"""
Dr.Case — Модуль відбору кандидатів (Candidate Selector)

Відбирає діагнози-кандидати на основі SOM проєкції.
Звужує простір пошуку з 842 діагнозів до ~14 кандидатів.

Компоненти:
- CandidateSelector: Головний клас для відбору
- SelectionResult: Результат відбору з усіма даними

Приклад використання:
    from dr_case.candidate_selector import CandidateSelector
    
    # Завантажити з файлу моделі
    selector = CandidateSelector.from_model_file(
        model_path="models/som_optimized.pkl",
        database_path="data/unified_disease_symptom_data_full.json"
    )
    
    # Відбір кандидатів
    result = selector.select(["fever", "headache", "cough", "fatigue"])
    
    print(f"Candidates ({result.candidate_count}):")
    for disease in result.candidates[:10]:
        print(f"  - {disease}")
    
    # З негативними симптомами
    result = selector.select(
        present_symptoms=["fever", "cough"],
        absent_symptoms=["rash", "vomiting"]
    )
    
    # З CaseRecord
    from dr_case.schemas import CaseRecord, Symptom
    
    case = CaseRecord(symptoms=[
        Symptom(name="fever"),
        Symptom(name="headache"),
    ])
    result = selector.select_from_case(case)
"""

from .selector import CandidateSelector, SelectionResult


__all__ = [
    "CandidateSelector",
    "SelectionResult",
]
