"""
Dr.Case — Модуль генерації питань (Question Engine)

Генерує уточнюючі питання для звуження діагнозу на основі
Information Gain — міри того, наскільки відповідь на питання
зменшить невизначеність.

Компоненти:
- QuestionGenerator: Головний клас для генерації питань
- InformationGainCalculator: Обчислення Information Gain
- QuestionStrategy: Стратегії вибору питань

Приклад використання:
    from dr_case.question_engine import QuestionGenerator, QuestionStrategy
    
    # Створення генератора
    generator = QuestionGenerator.from_database(
        "data/unified_disease_symptom_data_full.json",
        strategy=QuestionStrategy.HYBRID
    )
    
    # Генерація питань
    candidates = ["Influenza", "Common Cold", "Bronchitis", "Pneumonia"]
    known_symptoms = ["fever", "cough"]
    
    questions = generator.generate(
        candidates=candidates,
        known_present=known_symptoms,
        n_questions=3
    )
    
    for q in questions:
        print(f"Q: {q.text}")
        print(f"   IG: {q.information_gain:.4f}")
    
    # Пояснення питання
    explanation = generator.explain_question(questions[0], candidates)
    print(explanation)
"""

from .information_gain import (
    InformationGainCalculator,
    SymptomInformationGain
)
from .generator import (
    QuestionGenerator,
    QuestionStrategy,
    QuestionCandidate,
    SAFETY_CRITICAL_SYMPTOMS
)


__all__ = [
    # Information Gain
    "InformationGainCalculator",
    "SymptomInformationGain",
    
    # Generator
    "QuestionGenerator",
    "QuestionStrategy",
    "QuestionCandidate",
    "SAFETY_CRITICAL_SYMPTOMS",
]
