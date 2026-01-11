"""
Dr.Case — Обчислення Information Gain

Information Gain визначає наскільки питання про симптом
допоможе розрізнити діагнози-кандидати.

Формула:
    IG(symptom) = H(diseases) - H(diseases | symptom)
    
де H — ентропія розподілу.

Симптом з найвищим IG найкраще розділяє кандидатів на групи.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SymptomInformationGain:
    """Результат обчислення IG для симптому"""
    symptom: str
    information_gain: float
    
    # Розподіл кандидатів
    diseases_with_symptom: List[str]
    diseases_without_symptom: List[str]
    
    # Ймовірності
    p_present: float  # P(symptom=1)
    p_absent: float   # P(symptom=0)
    
    # Ентропії
    entropy_if_present: float
    entropy_if_absent: float
    
    @property
    def split_ratio(self) -> float:
        """Співвідношення розділення (ідеально = 0.5)"""
        total = len(self.diseases_with_symptom) + len(self.diseases_without_symptom)
        if total == 0:
            return 0.0
        return len(self.diseases_with_symptom) / total
    
    @property
    def is_balanced(self) -> bool:
        """Чи збалансоване розділення (30-70%)"""
        return 0.3 <= self.split_ratio <= 0.7


class InformationGainCalculator:
    """
    Калькулятор Information Gain для симптомів.
    
    Приклад використання:
        calculator = InformationGainCalculator(
            disease_symptom_matrix,
            disease_names,
            symptom_names
        )
        
        # Для списку кандидатів
        candidates = ["Influenza", "Common Cold", "Bronchitis"]
        known_symptoms = ["fever", "cough"]
        
        results = calculator.compute_all(
            candidates=candidates,
            known_present=known_symptoms
        )
        
        # Топ симптом для питання
        best = results[0]
        print(f"Ask about: {best.symptom} (IG={best.information_gain:.4f})")
    """
    
    def __init__(
        self,
        disease_symptom_matrix: np.ndarray,
        disease_names: List[str],
        symptom_names: List[str]
    ):
        """
        Args:
            disease_symptom_matrix: Матриця (N_diseases, N_symptoms), 1=має симптом
            disease_names: Список назв діагнозів
            symptom_names: Список назв симптомів
        """
        self.matrix = disease_symptom_matrix
        self.disease_names = disease_names
        self.symptom_names = symptom_names
        
        # Індекси
        self.disease_to_idx = {d: i for i, d in enumerate(disease_names)}
        self.symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}
        
        self.n_diseases = len(disease_names)
        self.n_symptoms = len(symptom_names)
    
    def _entropy(self, probabilities: np.ndarray) -> float:
        """
        Обчислити ентропію розподілу.
        
        H = -Σ p_i * log2(p_i)
        """
        # Фільтруємо нулі
        p = probabilities[probabilities > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))
    
    def _get_candidate_indices(self, candidates: List[str]) -> np.ndarray:
        """Отримати індекси кандидатів"""
        indices = []
        for disease in candidates:
            if disease in self.disease_to_idx:
                indices.append(self.disease_to_idx[disease])
        return np.array(indices)
    
    def compute_for_symptom(
        self,
        symptom: str,
        candidate_indices: np.ndarray,
        disease_weights: Optional[np.ndarray] = None
    ) -> SymptomInformationGain:
        """
        Обчислити Information Gain для одного симптому.
        
        Args:
            symptom: Назва симптому
            candidate_indices: Індекси кандидатів
            disease_weights: Ваги діагнозів (ймовірності)
            
        Returns:
            SymptomInformationGain
        """
        if symptom not in self.symptom_to_idx:
            return SymptomInformationGain(
                symptom=symptom,
                information_gain=0.0,
                diseases_with_symptom=[],
                diseases_without_symptom=[],
                p_present=0.0,
                p_absent=1.0,
                entropy_if_present=0.0,
                entropy_if_absent=0.0
            )
        
        symptom_idx = self.symptom_to_idx[symptom]
        n_candidates = len(candidate_indices)
        
        if n_candidates == 0:
            return SymptomInformationGain(
                symptom=symptom,
                information_gain=0.0,
                diseases_with_symptom=[],
                diseases_without_symptom=[],
                p_present=0.0,
                p_absent=1.0,
                entropy_if_present=0.0,
                entropy_if_absent=0.0
            )
        
        # Ваги за замовчуванням — рівномірні
        if disease_weights is None:
            disease_weights = np.ones(n_candidates) / n_candidates
        else:
            # Нормалізуємо
            disease_weights = disease_weights / disease_weights.sum()
        
        # Вектор симптому для кандидатів
        symptom_vector = self.matrix[candidate_indices, symptom_idx]
        
        # Розділяємо на групи
        has_symptom = symptom_vector > 0
        
        # Ймовірності P(symptom=1) та P(symptom=0)
        p_present = disease_weights[has_symptom].sum()
        p_absent = disease_weights[~has_symptom].sum()
        
        # Поточна ентропія H(diseases)
        current_entropy = self._entropy(disease_weights)
        
        # Ентропія якщо symptom=1
        if p_present > 0:
            weights_if_present = disease_weights[has_symptom] / p_present
            entropy_if_present = self._entropy(weights_if_present)
        else:
            entropy_if_present = 0.0
        
        # Ентропія якщо symptom=0
        if p_absent > 0:
            weights_if_absent = disease_weights[~has_symptom] / p_absent
            entropy_if_absent = self._entropy(weights_if_absent)
        else:
            entropy_if_absent = 0.0
        
        # Умовна ентропія H(diseases | symptom)
        conditional_entropy = p_present * entropy_if_present + p_absent * entropy_if_absent
        
        # Information Gain
        ig = current_entropy - conditional_entropy
        
        # Списки діагнозів
        diseases_with = [self.disease_names[i] for i in candidate_indices[has_symptom]]
        diseases_without = [self.disease_names[i] for i in candidate_indices[~has_symptom]]
        
        return SymptomInformationGain(
            symptom=symptom,
            information_gain=ig,
            diseases_with_symptom=diseases_with,
            diseases_without_symptom=diseases_without,
            p_present=p_present,
            p_absent=p_absent,
            entropy_if_present=entropy_if_present,
            entropy_if_absent=entropy_if_absent
        )
    
    def compute_all(
        self,
        candidates: List[str],
        known_present: Optional[List[str]] = None,
        known_absent: Optional[List[str]] = None,
        disease_weights: Optional[Dict[str, float]] = None,
        top_n: Optional[int] = None
    ) -> List[SymptomInformationGain]:
        """
        Обчислити Information Gain для всіх невідомих симптомів.
        
        Args:
            candidates: Список кандидатів
            known_present: Відомі присутні симптоми
            known_absent: Відомі відсутні симптоми
            disease_weights: Ваги діагнозів {disease: weight}
            top_n: Повернути тільки топ-N
            
        Returns:
            Список SymptomInformationGain, відсортований за IG (спадно)
        """
        known_present = set(known_present or [])
        known_absent = set(known_absent or [])
        known_symptoms = known_present | known_absent
        
        # Індекси кандидатів
        candidate_indices = self._get_candidate_indices(candidates)
        
        if len(candidate_indices) == 0:
            return []
        
        # Ваги
        if disease_weights:
            weights = np.array([
                disease_weights.get(self.disease_names[i], 1.0)
                for i in candidate_indices
            ])
        else:
            weights = None
        
        # Обчислюємо IG для кожного невідомого симптому
        results = []
        
        for symptom in self.symptom_names:
            if symptom in known_symptoms:
                continue
            
            ig_result = self.compute_for_symptom(symptom, candidate_indices, weights)
            
            # Пропускаємо симптоми з нульовим IG
            if ig_result.information_gain > 0:
                results.append(ig_result)
        
        # Сортуємо за IG (спадно)
        results.sort(key=lambda x: x.information_gain, reverse=True)
        
        if top_n:
            results = results[:top_n]
        
        return results
    
    def get_discriminative_symptoms(
        self,
        candidates: List[str],
        min_split_ratio: float = 0.2,
        max_split_ratio: float = 0.8
    ) -> List[str]:
        """
        Отримати симптоми, що добре розділяють кандидатів.
        
        Симптом є дискримінативним, якщо він присутній
        у частини кандидатів (не в усіх і не в жодному).
        
        Args:
            candidates: Список кандидатів
            min_split_ratio: Мінімальна частка з симптомом
            max_split_ratio: Максимальна частка з симптомом
            
        Returns:
            Список дискримінативних симптомів
        """
        candidate_indices = self._get_candidate_indices(candidates)
        n_candidates = len(candidate_indices)
        
        if n_candidates == 0:
            return []
        
        discriminative = []
        
        for symptom_idx, symptom in enumerate(self.symptom_names):
            # Частка кандидатів з цим симптомом
            symptom_vector = self.matrix[candidate_indices, symptom_idx]
            ratio = symptom_vector.sum() / n_candidates
            
            if min_split_ratio <= ratio <= max_split_ratio:
                discriminative.append(symptom)
        
        return discriminative
