"""
Dr.Case — Циклічний процес діагностики

Модулі:
- stopping_criteria: Критерії зупинки (DOMINANCE, STABILITY, NEED_TEST, SAFETY)
- hypothesis_tracker: Трекер гіпотез (тренди, виключення, модифікатори)
- cycle_controller: Головний контролер циклу
- feedback_processor: Обробка зворотного зв'язку

Головний клас:
    DiagnosisCycleController - об'єднує всі компоненти

Приклад використання:
    from dr_case.diagnosis_cycle import DiagnosisCycleController
    
    # Ініціалізація
    controller = DiagnosisCycleController.from_models(
        database_path="data/unified_disease_symptom_merged.json",
        som_path="models/som_merged.pkl",
        nn_path="models/nn_two_branch.pt"
    )
    
    # Автоматична діагностика
    def answer_func(question):
        print(f"Q: {question.text}")
        ans = input("(y/n/skip): ")
        return {'y': True, 'n': False}.get(ans, None)
    
    result = controller.run_full_diagnosis(
        initial_symptoms=['Fever', 'Cough'],
        answer_func=answer_func
    )
    
    print(f"Діагноз: {result.top_diagnosis} ({result.top_confidence:.1%})")
"""

from .stopping_criteria import (
    StopReason,
    StoppingConfig,
    StopDecision,
    StoppingCriteria
)

from .hypothesis_tracker import (
    HypothesisTrend,
    HypothesisSnapshot,
    HypothesisChange,
    HypothesisTracker
)

from .cycle_controller import (
    SessionPhase,
    DiagnosisQuestion,
    IterationResult,
    DiagnosisCycleResult,
    DiagnosisCycleController
)

from .feedback_processor import (
    FeedbackType,
    FeedbackConfig,
    Feedback,
    FeedbackResult,
    FeedbackProcessor
)


__all__ = [
    # Stopping Criteria
    'StopReason',
    'StoppingConfig', 
    'StopDecision',
    'StoppingCriteria',
    
    # Hypothesis Tracker
    'HypothesisTrend',
    'HypothesisSnapshot',
    'HypothesisChange',
    'HypothesisTracker',
    
    # Cycle Controller
    'SessionPhase',
    'DiagnosisQuestion',
    'IterationResult',
    'DiagnosisCycleResult',
    'DiagnosisCycleController',
    
    # Feedback Processor
    'FeedbackType',
    'FeedbackConfig',
    'Feedback',
    'FeedbackResult',
    'FeedbackProcessor',
]
