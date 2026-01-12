"""
Dr.Case — Критерії зупинки діагностики

Критерії:
- DOMINANCE: є чіткий лідер (ŷ_top > 0.85 AND gap > 0.3)
- STABILITY: гіпотези стабілізувались (3 ітерації без змін)
- NEED_TEST: потрібні лабораторні тести (топ гіпотези занадто близькі)
- SAFETY: досягнуто ліміту ітерацій або таймаут
- CONTINUE: продовжуємо діагностику
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class StopReason(Enum):
    """Причина зупинки діагностики"""
    CONTINUE = "continue"           # Продовжуємо
    DOMINANCE = "dominance"         # Є чіткий лідер
    STABILITY = "stability"         # Гіпотези стабільні
    NEED_TEST = "need_test"         # Потрібні лаб. тести
    SAFETY_LIMIT = "safety_limit"   # Досягнуто ліміту
    CONFIDENT = "confident"         # Висока впевненість
    NO_QUESTIONS = "no_questions"   # Немає більше питань


@dataclass
class StoppingConfig:
    """Конфігурація критеріїв зупинки"""
    
    # DOMINANCE
    dominance_threshold: float = 0.85     # ŷ_top > threshold
    dominance_gap: float = 0.30           # ŷ_top - ŷ_second > gap
    
    # STABILITY
    stability_iterations: int = 3         # Кількість стабільних ітерацій
    stability_tolerance: float = 0.05     # Допустима зміна
    
    # NEED_TEST
    need_test_top_n: int = 2              # Кількість топ гіпотез для порівняння
    need_test_threshold: float = 0.10     # Різниця < threshold → потрібні тести
    
    # SAFETY
    max_iterations: int = 20              # Максимум ітерацій
    max_questions: int = 30               # Максимум питань
    timeout_minutes: float = 30.0         # Таймаут (хвилини)
    
    # CONFIDENT
    confidence_threshold: float = 0.80    # Поріг впевненості


@dataclass
class StopDecision:
    """Результат перевірки критеріїв зупинки"""
    reason: StopReason
    should_stop: bool
    message: str = ""
    
    # Деталі для різних причин
    top_confidence: float = 0.0
    confidence_gap: float = 0.0
    stable_iterations: int = 0
    recommended_tests: List[str] = field(default_factory=list)
    
    @property
    def should_continue(self) -> bool:
        return not self.should_stop


class StoppingCriteria:
    """
    Перевірка критеріїв зупинки діагностики.
    
    Приклад:
        criteria = StoppingCriteria()
        
        decision = criteria.check(
            current_hypotheses={'Influenza': 0.75, 'Common Cold': 0.20},
            iteration=3,
            hypothesis_history=[...],
            elapsed_minutes=5.0
        )
        
        if decision.should_stop:
            print(f"Зупинка: {decision.reason.value}")
            print(f"Повідомлення: {decision.message}")
    """
    
    def __init__(self, config: Optional[StoppingConfig] = None):
        self.config = config or StoppingConfig()
    
    def check(
        self,
        current_hypotheses: Dict[str, float],
        iteration: int,
        questions_asked: int = 0,
        hypothesis_history: Optional[List[Dict[str, float]]] = None,
        elapsed_minutes: float = 0.0,
        available_questions: int = 1
    ) -> StopDecision:
        """
        Перевірити всі критерії зупинки.
        
        Args:
            current_hypotheses: Поточні гіпотези {disease: probability}
            iteration: Номер ітерації
            questions_asked: Кількість заданих питань
            hypothesis_history: Історія гіпотез (для STABILITY)
            elapsed_minutes: Час від початку (хвилини)
            available_questions: Кількість доступних питань
            
        Returns:
            StopDecision
        """
        if hypothesis_history is None:
            hypothesis_history = []
        
        # 1. Перевірка SAFETY (найвищий пріоритет)
        safety_decision = self._check_safety(
            iteration, questions_asked, elapsed_minutes
        )
        if safety_decision.should_stop:
            return safety_decision
        
        # 2. Перевірка NO_QUESTIONS
        if available_questions <= 0:
            return StopDecision(
                reason=StopReason.NO_QUESTIONS,
                should_stop=True,
                message="Немає більше доступних питань"
            )
        
        # 3. Перевірка DOMINANCE (є чіткий лідер)
        dominance_decision = self._check_dominance(current_hypotheses)
        if dominance_decision.should_stop:
            return dominance_decision
        
        # 4. Перевірка CONFIDENT (висока впевненість)
        confident_decision = self._check_confident(current_hypotheses)
        if confident_decision.should_stop:
            return confident_decision
        
        # 5. Перевірка STABILITY
        if len(hypothesis_history) >= self.config.stability_iterations:
            stability_decision = self._check_stability(
                current_hypotheses, hypothesis_history
            )
            if stability_decision.should_stop:
                return stability_decision
        
        # 6. Перевірка NEED_TEST (останній)
        need_test_decision = self._check_need_test(current_hypotheses)
        if need_test_decision.should_stop:
            return need_test_decision
        
        # Продовжуємо
        return StopDecision(
            reason=StopReason.CONTINUE,
            should_stop=False,
            message="Продовжуємо діагностику",
            top_confidence=self._get_top_confidence(current_hypotheses)
        )
    
    def _check_dominance(self, hypotheses: Dict[str, float]) -> StopDecision:
        """
        Перевірка DOMINANCE: ŷ_top > threshold AND gap > min_gap
        """
        if len(hypotheses) < 2:
            return StopDecision(
                reason=StopReason.CONTINUE,
                should_stop=False
            )
        
        sorted_hyp = sorted(hypotheses.items(), key=lambda x: x[1], reverse=True)
        top_name, top_conf = sorted_hyp[0]
        second_conf = sorted_hyp[1][1]
        gap = top_conf - second_conf
        
        if top_conf > self.config.dominance_threshold and gap > self.config.dominance_gap:
            return StopDecision(
                reason=StopReason.DOMINANCE,
                should_stop=True,
                message=f"Домінуючий діагноз: {top_name} ({top_conf:.1%}), "
                        f"відрив від другого: {gap:.1%}",
                top_confidence=top_conf,
                confidence_gap=gap
            )
        
        return StopDecision(
            reason=StopReason.CONTINUE,
            should_stop=False,
            top_confidence=top_conf,
            confidence_gap=gap
        )
    
    def _check_confident(self, hypotheses: Dict[str, float]) -> StopDecision:
        """
        Перевірка CONFIDENT: топ впевненість > threshold
        """
        if not hypotheses:
            return StopDecision(reason=StopReason.CONTINUE, should_stop=False)
        
        top_conf = max(hypotheses.values())
        
        if top_conf >= self.config.confidence_threshold:
            top_name = max(hypotheses, key=hypotheses.get)
            return StopDecision(
                reason=StopReason.CONFIDENT,
                should_stop=True,
                message=f"Висока впевненість в діагнозі: {top_name} ({top_conf:.1%})",
                top_confidence=top_conf
            )
        
        return StopDecision(
            reason=StopReason.CONTINUE,
            should_stop=False,
            top_confidence=top_conf
        )
    
    def _check_stability(
        self,
        current: Dict[str, float],
        history: List[Dict[str, float]]
    ) -> StopDecision:
        """
        Перевірка STABILITY: R^(t) ≈ R^(t-1) ≈ R^(t-2)
        """
        n = self.config.stability_iterations
        tolerance = self.config.stability_tolerance
        
        if len(history) < n - 1:
            return StopDecision(reason=StopReason.CONTINUE, should_stop=False)
        
        # Беремо останні n-1 з історії + поточний
        recent = history[-(n-1):] + [current]
        
        # Перевіряємо чи топ-3 гіпотези стабільні
        stable_count = 0
        
        for disease in current.keys():
            values = [h.get(disease, 0) for h in recent]
            if len(values) >= n:
                max_val = max(values)
                min_val = min(values)
                if max_val - min_val < tolerance:
                    stable_count += 1
        
        if stable_count >= 3:  # Мінімум 3 стабільних гіпотези
            return StopDecision(
                reason=StopReason.STABILITY,
                should_stop=True,
                message=f"Гіпотези стабільні протягом {n} ітерацій "
                        f"({stable_count} стабільних діагнозів)",
                stable_iterations=n
            )
        
        return StopDecision(
            reason=StopReason.CONTINUE,
            should_stop=False,
            stable_iterations=stable_count
        )
    
    def _check_need_test(self, hypotheses: Dict[str, float]) -> StopDecision:
        """
        Перевірка NEED_TEST: топ гіпотези занадто близькі
        
        Спрацьовує тільки якщо:
        1. Топ-2 гіпотези мають різницю < threshold
        2. Топ гіпотеза має достатню впевненість (> 30%)
        """
        if len(hypotheses) < 2:
            return StopDecision(reason=StopReason.CONTINUE, should_stop=False)
        
        sorted_hyp = sorted(hypotheses.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_hyp[:self.config.need_test_top_n]
        
        # Перевіряємо чи топ має достатню впевненість
        top_conf = top_n[0][1]
        if top_conf < 0.30:  # Якщо топ < 30%, ще рано для тестів
            return StopDecision(reason=StopReason.CONTINUE, should_stop=False)
        
        # Перевіряємо різницю між топ-N
        values = [h[1] for h in top_n]
        diff = max(values) - min(values)
        
        if diff < self.config.need_test_threshold:
            diseases = [h[0] for h in top_n]
            return StopDecision(
                reason=StopReason.NEED_TEST,
                should_stop=True,
                message=f"Потрібні лабораторні тести для розрізнення: "
                        f"{', '.join(diseases)} (різниця лише {diff:.1%})",
                recommended_tests=[]  # Буде заповнено LabRecommender
            )
        
        return StopDecision(reason=StopReason.CONTINUE, should_stop=False)
    
    def _check_safety(
        self,
        iteration: int,
        questions_asked: int,
        elapsed_minutes: float
    ) -> StopDecision:
        """
        Перевірка SAFETY: ліміти безпеки
        """
        # Ліміт ітерацій
        if iteration >= self.config.max_iterations:
            return StopDecision(
                reason=StopReason.SAFETY_LIMIT,
                should_stop=True,
                message=f"Досягнуто ліміту ітерацій ({self.config.max_iterations})"
            )
        
        # Ліміт питань
        if questions_asked >= self.config.max_questions:
            return StopDecision(
                reason=StopReason.SAFETY_LIMIT,
                should_stop=True,
                message=f"Досягнуто ліміту питань ({self.config.max_questions})"
            )
        
        # Таймаут
        if elapsed_minutes >= self.config.timeout_minutes:
            return StopDecision(
                reason=StopReason.SAFETY_LIMIT,
                should_stop=True,
                message=f"Таймаут ({self.config.timeout_minutes:.0f} хв)"
            )
        
        return StopDecision(reason=StopReason.CONTINUE, should_stop=False)
    
    def _get_top_confidence(self, hypotheses: Dict[str, float]) -> float:
        """Отримати впевненість топ гіпотези"""
        return max(hypotheses.values()) if hypotheses else 0.0
    
    def __repr__(self) -> str:
        return (
            f"StoppingCriteria("
            f"dominance={self.config.dominance_threshold:.0%}, "
            f"stability={self.config.stability_iterations} iter, "
            f"max_iter={self.config.max_iterations})"
        )
