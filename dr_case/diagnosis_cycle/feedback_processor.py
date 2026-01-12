"""
Dr.Case — Обробка зворотного зв'язку

Типи feedback:
- TREATMENT_FAILED: лікування не допомогло
- NEW_SYMPTOM: з'явився новий симптом
- CONDITION_CHANGED: стан пацієнта змінився
- DOCTOR_OVERRIDE: лікар має іншу думку

При невдачі лікування:
1. Знизити впевненість у поточному діагнозі
2. Повернути виключені гіпотези
3. Розширити пошук (α, k)
4. Перезапустити цикл
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FeedbackType(Enum):
    """Тип зворотного зв'язку"""
    TREATMENT_SUCCESS = "treatment_success"   # Лікування допомогло
    TREATMENT_FAILED = "treatment_failed"     # Лікування не допомогло
    NEW_SYMPTOM = "new_symptom"               # Новий симптом
    SYMPTOM_RESOLVED = "symptom_resolved"     # Симптом зник
    CONDITION_WORSENED = "condition_worsened" # Стан погіршився
    CONDITION_IMPROVED = "condition_improved" # Стан покращився
    DOCTOR_OVERRIDE = "doctor_override"       # Лікар поставив інший діагноз
    TEST_RESULT = "test_result"               # Результат аналізу


@dataclass
class FeedbackConfig:
    """Конфігурація обробки зворотного зв'язку"""
    
    # При невдачі лікування
    treatment_failure_downgrade: float = 0.3    # Знизити на 70%
    alternatives_boost: float = 1.5             # Підняти альтернативи на 50%
    
    # Розширення пошуку
    expanded_alpha: float = 0.98                # Розширений α
    expanded_k: int = 15                        # Розширений k
    
    # Ліміти
    max_restart_attempts: int = 3               # Максимум перезапусків


@dataclass
class Feedback:
    """Зворотний зв'язок від користувача/лікаря"""
    type: FeedbackType
    
    # Для TREATMENT_FAILED
    failed_diagnosis: str = ""
    
    # Для NEW_SYMPTOM / SYMPTOM_RESOLVED
    symptom: str = ""
    
    # Для TEST_RESULT
    test_name: str = ""
    test_result: str = ""
    test_values: Dict[str, Any] = field(default_factory=dict)
    
    # Для DOCTOR_OVERRIDE
    doctor_diagnosis: str = ""
    doctor_notes: str = ""
    
    # Метадані
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "user"  # user, doctor, system


@dataclass 
class FeedbackResult:
    """Результат обробки зворотного зв'язку"""
    action_taken: str
    should_restart: bool = False
    modified_hypotheses: Dict[str, float] = field(default_factory=dict)
    restored_hypotheses: List[str] = field(default_factory=list)
    new_symptoms: List[str] = field(default_factory=list)
    message: str = ""


class FeedbackProcessor:
    """
    Обробник зворотного зв'язку.
    
    Приклад:
        processor = FeedbackProcessor()
        
        # При невдачі лікування
        feedback = Feedback(
            type=FeedbackType.TREATMENT_FAILED,
            failed_diagnosis="Influenza"
        )
        
        result = processor.process(feedback, hypothesis_tracker, session_state)
        
        if result.should_restart:
            # Перезапустити діагностику з оновленим станом
            controller.restart_with_feedback(result)
    """
    
    def __init__(self, config: Optional[FeedbackConfig] = None):
        self.config = config or FeedbackConfig()
        self.feedback_history: List[Feedback] = []
        self.restart_count = 0
    
    def process(
        self,
        feedback: Feedback,
        hypothesis_tracker: Any,  # HypothesisTracker
        session_state: Any = None  # SessionState
    ) -> FeedbackResult:
        """
        Обробити зворотний зв'язок.
        
        Args:
            feedback: Об'єкт зворотного зв'язку
            hypothesis_tracker: Трекер гіпотез
            session_state: Стан сесії (опціонально)
            
        Returns:
            FeedbackResult
        """
        self.feedback_history.append(feedback)
        
        if feedback.type == FeedbackType.TREATMENT_SUCCESS:
            return self._handle_treatment_success(feedback)
        
        elif feedback.type == FeedbackType.TREATMENT_FAILED:
            return self._handle_treatment_failed(
                feedback, hypothesis_tracker, session_state
            )
        
        elif feedback.type == FeedbackType.NEW_SYMPTOM:
            return self._handle_new_symptom(feedback, session_state)
        
        elif feedback.type == FeedbackType.SYMPTOM_RESOLVED:
            return self._handle_symptom_resolved(feedback, session_state)
        
        elif feedback.type == FeedbackType.CONDITION_WORSENED:
            return self._handle_condition_worsened(
                feedback, hypothesis_tracker, session_state
            )
        
        elif feedback.type == FeedbackType.CONDITION_IMPROVED:
            return self._handle_condition_improved(feedback)
        
        elif feedback.type == FeedbackType.DOCTOR_OVERRIDE:
            return self._handle_doctor_override(feedback)
        
        elif feedback.type == FeedbackType.TEST_RESULT:
            return self._handle_test_result(
                feedback, hypothesis_tracker, session_state
            )
        
        return FeedbackResult(
            action_taken="unknown_feedback_type",
            message=f"Невідомий тип зворотного зв'язку: {feedback.type}"
        )
    
    def _handle_treatment_success(self, feedback: Feedback) -> FeedbackResult:
        """Лікування допомогло — закриваємо кейс"""
        return FeedbackResult(
            action_taken="case_closed",
            should_restart=False,
            message="Лікування успішне. Кейс закрито."
        )
    
    def _handle_treatment_failed(
        self,
        feedback: Feedback,
        hypothesis_tracker: Any,
        session_state: Any
    ) -> FeedbackResult:
        """
        Лікування не допомогло — ключовий момент циклічності!
        
        Алгоритм:
        1. Знизити впевненість у поточному діагнозі
        2. Додати факт невдачі як негативний симптом
        3. Підняти альтернативні гіпотези
        4. Повернути виключені гіпотези
        5. Розширити пошук
        6. Перезапустити цикл
        """
        # Перевіряємо ліміт перезапусків
        if self.restart_count >= self.config.max_restart_attempts:
            return FeedbackResult(
                action_taken="max_restarts_reached",
                should_restart=False,
                message=f"Досягнуто ліміт перезапусків ({self.config.max_restart_attempts}). "
                        "Рекомендовано консультацію спеціаліста."
            )
        
        self.restart_count += 1
        failed_diagnosis = feedback.failed_diagnosis
        
        # 1. Знизити впевненість у діагнозі
        hypothesis_tracker.downgrade(
            failed_diagnosis, 
            factor=self.config.treatment_failure_downgrade
        )
        
        # 2. Виключити діагноз з подальшого розгляду
        hypothesis_tracker.exclude(
            failed_diagnosis,
            reason=f"Лікування не допомогло (спроба {self.restart_count})"
        )
        
        # 3. Підняти альтернативи
        current = hypothesis_tracker.get_current()
        alternatives = self._get_alternatives(current, failed_diagnosis)
        
        for alt in alternatives:
            hypothesis_tracker.boost(alt, factor=self.config.alternatives_boost)
        
        # 4. Повернути раніше виключені (крім поточного)
        restored = []
        for disease in list(hypothesis_tracker.excluded):
            if disease != failed_diagnosis:
                hypothesis_tracker.excluded.remove(disease)
                restored.append(disease)
        
        # 5. Додати негативний симптом (якщо є session_state)
        negative_symptom = f"no_response_to_{failed_diagnosis}_treatment"
        new_symptoms = []
        if session_state is not None:
            session_state.add_symptom(negative_symptom, present=True)
            new_symptoms.append(negative_symptom)
        
        return FeedbackResult(
            action_taken="restart_diagnosis",
            should_restart=True,
            modified_hypotheses=hypothesis_tracker.get_current(),
            restored_hypotheses=restored,
            new_symptoms=new_symptoms,
            message=f"Діагноз '{failed_diagnosis}' знижено. "
                    f"Відновлено {len(restored)} альтернативних гіпотез. "
                    f"Перезапуск діагностики..."
        )
    
    def _handle_new_symptom(
        self,
        feedback: Feedback,
        session_state: Any
    ) -> FeedbackResult:
        """Новий симптом — продовжуємо діагностику"""
        new_symptoms = [feedback.symptom] if feedback.symptom else []
        
        if session_state is not None and feedback.symptom:
            session_state.add_symptom(feedback.symptom, present=True)
        
        return FeedbackResult(
            action_taken="continue_with_new_symptom",
            should_restart=True,
            new_symptoms=new_symptoms,
            message=f"Додано новий симптом: '{feedback.symptom}'. "
                    "Продовження діагностики..."
        )
    
    def _handle_symptom_resolved(
        self,
        feedback: Feedback,
        session_state: Any
    ) -> FeedbackResult:
        """Симптом зник"""
        if session_state is not None and feedback.symptom:
            # Переводимо в негативний
            session_state.add_symptom(feedback.symptom, present=False)
        
        return FeedbackResult(
            action_taken="symptom_resolved",
            should_restart=True,
            message=f"Симптом '{feedback.symptom}' зник. "
                    "Оновлення діагностики..."
        )
    
    def _handle_condition_worsened(
        self,
        feedback: Feedback,
        hypothesis_tracker: Any,
        session_state: Any
    ) -> FeedbackResult:
        """Стан погіршився — терміново"""
        # Схоже на невдачу лікування, але серйозніше
        return FeedbackResult(
            action_taken="urgent_reassessment",
            should_restart=True,
            message="УВАГА: Стан пацієнта погіршився! "
                    "Рекомендовано термінову консультацію лікаря."
        )
    
    def _handle_condition_improved(self, feedback: Feedback) -> FeedbackResult:
        """Стан покращився"""
        return FeedbackResult(
            action_taken="continue_monitoring",
            should_restart=False,
            message="Стан пацієнта покращився. Продовжити моніторинг."
        )
    
    def _handle_doctor_override(self, feedback: Feedback) -> FeedbackResult:
        """Лікар поставив інший діагноз"""
        return FeedbackResult(
            action_taken="doctor_diagnosis_recorded",
            should_restart=False,
            message=f"Діагноз лікаря: '{feedback.doctor_diagnosis}'. "
                    "Збережено для навчання системи."
        )
    
    def _handle_test_result(
        self,
        feedback: Feedback,
        hypothesis_tracker: Any,
        session_state: Any
    ) -> FeedbackResult:
        """Результат лабораторного тесту"""
        # Додаємо результат тесту як симптом
        test_symptom = f"{feedback.test_name}_{feedback.test_result}"
        
        if session_state is not None:
            session_state.add_symptom(test_symptom, present=True)
        
        return FeedbackResult(
            action_taken="test_result_incorporated",
            should_restart=True,
            new_symptoms=[test_symptom],
            message=f"Результат тесту '{feedback.test_name}': {feedback.test_result}. "
                    "Оновлення діагностики..."
        )
    
    def _get_alternatives(
        self,
        current_hypotheses: Dict[str, float],
        failed_diagnosis: str
    ) -> List[str]:
        """Отримати альтернативні діагнози"""
        # Всі крім failed, з ймовірністю > 5%
        alternatives = []
        for disease, prob in current_hypotheses.items():
            if disease != failed_diagnosis and prob > 0.05:
                alternatives.append(disease)
        
        # Сортуємо за ймовірністю
        alternatives.sort(key=lambda d: current_hypotheses[d], reverse=True)
        
        return alternatives[:5]  # Топ-5 альтернатив
    
    def reset(self):
        """Скинути лічильники"""
        self.restart_count = 0
        self.feedback_history.clear()
    
    def __repr__(self) -> str:
        return (
            f"FeedbackProcessor("
            f"restarts={self.restart_count}, "
            f"history={len(self.feedback_history)})"
        )
