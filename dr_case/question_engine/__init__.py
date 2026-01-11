"""
Dr.Case — Модуль Question Engine

Генерує уточнюючі питання для звуження діагнозу на основі
Expected Information Gain (EIG).

EIG(q) = H(ŷ) - E[H(ŷ|answer)]

Компоненти:
- InformationGainCalculator: Обчислення EIG для симптомів
- QuestionSelector: Вибір найкращого питання
- AnswerProcessor: Обробка відповідей та оновлення стану

Приклад використання:
    from dr_case.question_engine import QuestionSelector, AnswerProcessor, AnswerType
    
    # Створення selector
    selector = QuestionSelector.from_database("data/database.json")
    
    # Поточні ймовірності від NN
    probs = {"Influenza": 0.45, "Cold": 0.32, "COVID": 0.15}
    known = {"Fever", "Cough"}
    
    # Вибір питання
    question = selector.select_question(probs, known_symptoms=known)
    
    if question:
        print(f"Питання: {question.text}")
        print(f"EIG: {question.eig:.4f}")
        
        # Обробка відповіді
        new_probs = selector.process_answer(probs, question.symptom, AnswerType.YES)
"""

from .information_gain import (
    InformationGainCalculator,
    EIGResult,
    AnswerType,
    FuzzyMembershipHandler,
    entropy,
    normalize_probs,
)

from .question_selector import (
    QuestionSelector,
    QuestionGenerator,
    Question,
    QuestionResult,
)

from .answer_processor import (
    AnswerProcessor,
    SessionState,
    SymptomState,
    SymptomVectorUpdater,
)


__all__ = [
    # Information Gain
    "InformationGainCalculator",
    "EIGResult",
    "AnswerType",
    "FuzzyMembershipHandler",
    "entropy",
    "normalize_probs",
    
    # Question Selector
    "QuestionSelector",
    "QuestionGenerator", 
    "Question",
    "QuestionResult",
    
    # Answer Processor
    "AnswerProcessor",
    "SessionState",
    "SymptomState",
    "SymptomVectorUpdater",
]
