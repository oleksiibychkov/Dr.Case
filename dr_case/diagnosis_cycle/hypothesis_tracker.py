"""
Dr.Case — Трекер гіпотез

Відстеження динаміки гіпотез між ітераціями:
- Зростаючі гіпотези
- Падаючі гіпотези
- Стабільні гіпотези
- Виключені гіпотези (при невдалому лікуванні)
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class HypothesisTrend(Enum):
    """Тренд гіпотези"""
    RISING = "rising"       # Зростає
    FALLING = "falling"     # Падає
    STABLE = "stable"       # Стабільна
    NEW = "new"             # Нова
    EXCLUDED = "excluded"   # Виключена


@dataclass
class HypothesisSnapshot:
    """Знімок гіпотез на ітерації"""
    iteration: int
    timestamp: datetime
    hypotheses: Dict[str, float]  # {disease: probability}
    
    def get_top_n(self, n: int = 5) -> List[Tuple[str, float]]:
        """Топ-N гіпотез"""
        sorted_hyp = sorted(
            self.hypotheses.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_hyp[:n]


@dataclass
class HypothesisChange:
    """Зміна гіпотези"""
    disease: str
    old_probability: float
    new_probability: float
    trend: HypothesisTrend
    
    @property
    def change(self) -> float:
        """Абсолютна зміна"""
        return self.new_probability - self.old_probability
    
    @property
    def change_percent(self) -> float:
        """Відносна зміна (%)"""
        if self.old_probability == 0:
            return float('inf') if self.new_probability > 0 else 0
        return (self.new_probability - self.old_probability) / self.old_probability * 100


class HypothesisTracker:
    """
    Трекер гіпотез для відстеження динаміки.
    
    Приклад:
        tracker = HypothesisTracker()
        
        # Оновлення на кожній ітерації
        tracker.update({'Influenza': 0.5, 'Cold': 0.3}, iteration=1)
        tracker.update({'Influenza': 0.6, 'Cold': 0.25}, iteration=2)
        
        # Отримання трендів
        rising = tracker.get_rising_hypotheses()
        falling = tracker.get_falling_hypotheses()
        
        # Виключення при невдалому лікуванні
        tracker.exclude('Influenza', reason='Treatment failed')
        
        # Відновлення при повторній діагностиці
        tracker.restore_excluded()
    """
    
    def __init__(self, trend_window: int = 3, tolerance: float = 0.05):
        """
        Args:
            trend_window: Кількість ітерацій для визначення тренду
            tolerance: Допустима зміна для "стабільності"
        """
        self.trend_window = trend_window
        self.tolerance = tolerance
        
        # Історія знімків
        self.history: List[HypothesisSnapshot] = []
        
        # Виключені гіпотези
        self.excluded: Set[str] = set()
        self.exclusion_reasons: Dict[str, str] = {}
        self.exclusion_times: Dict[str, datetime] = {}
        
        # Модифікатори (для boost/downgrade)
        self.modifiers: Dict[str, float] = {}  # {disease: multiplier}
    
    def update(
        self,
        hypotheses: Dict[str, float],
        iteration: int
    ) -> List[HypothesisChange]:
        """
        Оновити стан гіпотез.
        
        Args:
            hypotheses: Поточні гіпотези {disease: probability}
            iteration: Номер ітерації
            
        Returns:
            Список змін порівняно з попередньою ітерацією
        """
        # Застосовуємо модифікатори
        modified = self._apply_modifiers(hypotheses)
        
        # Фільтруємо виключені
        filtered = {d: p for d, p in modified.items() if d not in self.excluded}
        
        # Нормалізуємо (сума = 1)
        total = sum(filtered.values())
        if total > 0:
            filtered = {d: p / total for d, p in filtered.items()}
        
        # Створюємо знімок
        snapshot = HypothesisSnapshot(
            iteration=iteration,
            timestamp=datetime.now(),
            hypotheses=filtered
        )
        
        # Обчислюємо зміни
        changes = []
        if self.history:
            prev = self.history[-1].hypotheses
            changes = self._compute_changes(prev, filtered)
        
        self.history.append(snapshot)
        
        return changes
    
    def _apply_modifiers(self, hypotheses: Dict[str, float]) -> Dict[str, float]:
        """Застосувати модифікатори до гіпотез"""
        modified = {}
        for disease, prob in hypotheses.items():
            modifier = self.modifiers.get(disease, 1.0)
            modified[disease] = min(1.0, prob * modifier)
        return modified
    
    def _compute_changes(
        self,
        old: Dict[str, float],
        new: Dict[str, float]
    ) -> List[HypothesisChange]:
        """Обчислити зміни між двома знімками"""
        changes = []
        
        all_diseases = set(old.keys()) | set(new.keys())
        
        for disease in all_diseases:
            old_prob = old.get(disease, 0)
            new_prob = new.get(disease, 0)
            
            # Визначаємо тренд
            diff = new_prob - old_prob
            
            if disease not in old:
                trend = HypothesisTrend.NEW
            elif disease in self.excluded:
                trend = HypothesisTrend.EXCLUDED
            elif abs(diff) < self.tolerance:
                trend = HypothesisTrend.STABLE
            elif diff > 0:
                trend = HypothesisTrend.RISING
            else:
                trend = HypothesisTrend.FALLING
            
            changes.append(HypothesisChange(
                disease=disease,
                old_probability=old_prob,
                new_probability=new_prob,
                trend=trend
            ))
        
        return sorted(changes, key=lambda x: x.new_probability, reverse=True)
    
    def get_current(self) -> Dict[str, float]:
        """Отримати поточні гіпотези"""
        if not self.history:
            return {}
        return self.history[-1].hypotheses.copy()
    
    def get_top_hypotheses(self, n: int = 5) -> List[Tuple[str, float]]:
        """Отримати топ-N гіпотез"""
        if not self.history:
            return []
        return self.history[-1].get_top_n(n)
    
    def get_rising_hypotheses(self, n_last: int = None) -> List[str]:
        """
        Гіпотези, що зростають останні N ітерацій.
        
        Args:
            n_last: Кількість ітерацій (default: trend_window)
            
        Returns:
            Список зростаючих гіпотез
        """
        n = n_last or self.trend_window
        
        if len(self.history) < n:
            return []
        
        rising = []
        current = self.history[-1].hypotheses
        
        for disease in current.keys():
            values = [
                h.hypotheses.get(disease, 0) 
                for h in self.history[-n:]
            ]
            
            # Перевіряємо чи всі переходи зростаючі
            if all(values[i] < values[i+1] - 0.001 for i in range(len(values)-1)):
                rising.append(disease)
        
        return rising
    
    def get_falling_hypotheses(self, n_last: int = None) -> List[str]:
        """
        Гіпотези, що падають останні N ітерацій.
        
        Args:
            n_last: Кількість ітерацій (default: trend_window)
            
        Returns:
            Список падаючих гіпотез
        """
        n = n_last or self.trend_window
        
        if len(self.history) < n:
            return []
        
        falling = []
        current = self.history[-1].hypotheses
        
        for disease in current.keys():
            values = [
                h.hypotheses.get(disease, 0) 
                for h in self.history[-n:]
            ]
            
            # Перевіряємо чи всі переходи падаючі
            if all(values[i] > values[i+1] + 0.001 for i in range(len(values)-1)):
                falling.append(disease)
        
        return falling
    
    def get_stable_hypotheses(
        self,
        n_iterations: int = None,
        tolerance: float = None
    ) -> List[str]:
        """
        Гіпотези, стабільні N ітерацій.
        
        Args:
            n_iterations: Кількість ітерацій (default: trend_window)
            tolerance: Допустима зміна (default: self.tolerance)
            
        Returns:
            Список стабільних гіпотез
        """
        n = n_iterations or self.trend_window
        tol = tolerance or self.tolerance
        
        if len(self.history) < n:
            return []
        
        stable = []
        current = self.history[-1].hypotheses
        
        for disease in current.keys():
            values = [
                h.hypotheses.get(disease, 0) 
                for h in self.history[-n:]
            ]
            
            if max(values) - min(values) < tol:
                stable.append(disease)
        
        return stable
    
    def exclude(self, disease: str, reason: str = "") -> None:
        """
        Виключити гіпотезу.
        
        Args:
            disease: Назва хвороби
            reason: Причина виключення
        """
        self.excluded.add(disease)
        self.exclusion_reasons[disease] = reason
        self.exclusion_times[disease] = datetime.now()
    
    def restore_excluded(self) -> List[str]:
        """
        Повернути виключені гіпотези.
        
        Returns:
            Список відновлених гіпотез
        """
        restored = list(self.excluded)
        self.excluded.clear()
        self.exclusion_reasons.clear()
        self.exclusion_times.clear()
        return restored
    
    def downgrade(self, disease: str, factor: float = 0.5) -> None:
        """
        Знизити впевненість у діагнозі.
        
        Args:
            disease: Назва хвороби
            factor: Множник (0.5 = зменшити вдвічі)
        """
        current_modifier = self.modifiers.get(disease, 1.0)
        self.modifiers[disease] = current_modifier * factor
    
    def boost(self, disease: str, factor: float = 1.2) -> None:
        """
        Підвищити впевненість у діагнозі.
        
        Args:
            disease: Назва хвороби
            factor: Множник (1.2 = збільшити на 20%)
        """
        current_modifier = self.modifiers.get(disease, 1.0)
        self.modifiers[disease] = min(2.0, current_modifier * factor)
    
    def reset_modifiers(self) -> None:
        """Скинути всі модифікатори"""
        self.modifiers.clear()
    
    def get_history_for_disease(self, disease: str) -> List[Tuple[int, float]]:
        """
        Отримати історію ймовірності для хвороби.
        
        Args:
            disease: Назва хвороби
            
        Returns:
            Список (iteration, probability)
        """
        return [
            (h.iteration, h.hypotheses.get(disease, 0))
            for h in self.history
        ]
    
    def get_hypothesis_history(self) -> List[Dict[str, float]]:
        """Отримати історію гіпотез для StoppingCriteria"""
        return [h.hypotheses for h in self.history]
    
    @property
    def n_iterations(self) -> int:
        """Кількість ітерацій"""
        return len(self.history)
    
    @property
    def n_excluded(self) -> int:
        """Кількість виключених гіпотез"""
        return len(self.excluded)
    
    def __repr__(self) -> str:
        current = self.get_current()
        top = max(current, key=current.get) if current else "None"
        return (
            f"HypothesisTracker("
            f"iterations={self.n_iterations}, "
            f"top={top}, "
            f"excluded={self.n_excluded})"
        )
