"""
Dr.Case — Проєкція на SOM та обчислення membership

Ключовий компонент для Candidate Selector:
- Проєкція вектора пацієнта на SOM
- Обчислення membership (softmax від відстаней)
- Відбір активних юнітів за параметрами (α, k, τ)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from dr_case.config import CandidateSelectorConfig
from dr_case.schemas import SOMResult
from .model import SOMModel


@dataclass
class ProjectionResult:
    """Результат проєкції пацієнта на SOM"""
    # BMU
    bmu: Tuple[int, int]
    bmu_distance: float
    
    # Всі відстані
    distances: np.ndarray  # shape (H, W)
    
    # Membership
    memberships: Dict[Tuple[int, int], float]
    
    # Активні юніти (за критеріями α, k, τ)
    active_units: List[Tuple[int, int]]
    
    # Кандидати (діагнози з активних юнітів)
    candidate_diseases: List[str]
    
    # Статистика
    cumulative_mass: float
    
    def to_som_result(self) -> SOMResult:
        """Конвертувати в SOMResult для schemas"""
        memberships_str = {
            f"{u[0]}_{u[1]}": m 
            for u, m in self.memberships.items()
        }
        
        top_units_str = [f"{u[0]}_{u[1]}" for u in self.active_units]
        
        return SOMResult(
            bmu_coords=self.bmu,
            bmu_distance=self.bmu_distance,
            memberships=memberships_str,
            top_units=top_units_str,
            active_units_count=len(self.active_units),
            cumulative_mass=self.cumulative_mass
        )


class SOMProjector:
    """
    Проєктор пацієнта на SOM з обчисленням membership.
    
    Membership обчислюється як softmax від негативних відстаней:
    m_i = exp(-λ * d_i) / Σ exp(-λ * d_j)
    
    Відбір активних юнітів за критеріями:
    - α: cumulative mass (наприклад, 0.9 = 90% маси)
    - k: максимальна кількість юнітів
    - τ: мінімальний поріг membership
    
    Приклад використання:
        projector = SOMProjector(som_model, selector_config)
        result = projector.project(patient_vector)
        
        print(f"BMU: {result.bmu}")
        print(f"Candidates: {result.candidate_diseases}")
    """
    
    def __init__(
        self, 
        som_model: SOMModel, 
        config: Optional[CandidateSelectorConfig] = None
    ):
        """
        Args:
            som_model: Навчена модель SOM
            config: Конфігурація Candidate Selector
        """
        self.som = som_model
        self.config = config or CandidateSelectorConfig()
    
    def compute_memberships(
        self, 
        distances: np.ndarray,
        lambda_param: Optional[float] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Обчислити membership для всіх юнітів.
        
        Формула: m_i = exp(-λ * d_i) / Σ exp(-λ * d_j)
        
        Тільки для заповнених юнітів (що мають діагнози).
        
        Args:
            distances: Матриця відстаней shape (H, W)
            lambda_param: Параметр гостроти softmax
            
        Returns:
            Словник {(row, col): membership}
        """
        if lambda_param is None:
            lambda_param = self.config.membership_lambda
        
        # Беремо тільки заповнені юніти
        filled_units = self.som.filled_units
        
        if not filled_units:
            return {}
        
        # Збираємо відстані для заповнених юнітів
        filled_distances = []
        filled_coords = []
        
        for unit in filled_units:
            filled_distances.append(distances[unit[0], unit[1]])
            filled_coords.append(unit)
        
        filled_distances = np.array(filled_distances)
        
        # Softmax з негативними відстанями
        # Для числової стабільності віднімаємо мінімум
        neg_distances = -lambda_param * filled_distances
        neg_distances -= np.max(neg_distances)  # стабільність
        
        exp_values = np.exp(neg_distances)
        softmax_values = exp_values / np.sum(exp_values)
        
        # Формуємо словник
        memberships = {}
        for i, unit in enumerate(filled_coords):
            memberships[unit] = float(softmax_values[i])
        
        return memberships
    
    def select_active_units(
        self,
        memberships: Dict[Tuple[int, int], float],
        alpha: Optional[float] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Відібрати активні юніти за критеріями.
        
        Критерії (застосовуються послідовно):
        1. Сортуємо юніти за membership (спадно)
        2. Беремо поки cumulative mass < α
        3. Але не більше k юнітів
        4. І тільки якщо membership > τ
        
        Args:
            memberships: Словник membership
            alpha: Cumulative mass threshold
            k: Max кількість юнітів
            tau: Мінімальний поріг membership
            
        Returns:
            (список активних юнітів, досягнутий cumulative mass)
        """
        if alpha is None:
            alpha = self.config.alpha
        if k is None:
            k = self.config.k
        if tau is None:
            tau = self.config.tau
        
        if not memberships:
            return [], 0.0
        
        # Сортуємо за membership (спадно)
        sorted_units = sorted(
            memberships.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        active_units = []
        cumulative_mass = 0.0
        
        for unit, membership in sorted_units:
            # Перевіряємо поріг τ
            if membership < tau:
                break
            
            # Перевіряємо ліміт k
            if len(active_units) >= k:
                break
            
            active_units.append(unit)
            cumulative_mass += membership
            
            # Перевіряємо cumulative mass α
            if cumulative_mass >= alpha:
                break
        
        return active_units, cumulative_mass
    
    def get_candidate_diseases(
        self, 
        active_units: List[Tuple[int, int]]
    ) -> List[str]:
        """
        Отримати діагнози-кандидати з активних юнітів.
        
        Args:
            active_units: Список активних юнітів
            
        Returns:
            Список унікальних назв діагнозів
        """
        candidates = []
        seen = set()
        
        for unit in active_units:
            diseases = self.som.get_diseases_in_unit(unit)
            for disease in diseases:
                if disease not in seen:
                    candidates.append(disease)
                    seen.add(disease)
        
        return candidates
    
    def project(
        self, 
        vector: np.ndarray,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None
    ) -> ProjectionResult:
        """
        Проєктувати вектор пацієнта на SOM.
        
        Повний pipeline:
        1. Знайти BMU
        2. Обчислити всі відстані
        3. Обчислити membership (softmax)
        4. Відібрати активні юніти (α, k, τ)
        5. Зібрати кандидатів
        
        Args:
            vector: Вектор симптомів пацієнта shape (D,)
            alpha, k, tau: Override параметрів
            
        Returns:
            ProjectionResult з усіма даними
        """
        # 1. BMU
        bmu = self.som.get_bmu(vector)
        bmu_distance = self.som.get_bmu_distance(vector)
        
        # 2. Всі відстані
        distances = self.som.get_all_distances(vector)
        
        # 3. Membership
        memberships = self.compute_memberships(distances)
        
        # 4. Активні юніти
        active_units, cumulative_mass = self.select_active_units(
            memberships, alpha, k, tau
        )
        
        # 5. Кандидати
        candidate_diseases = self.get_candidate_diseases(active_units)
        
        return ProjectionResult(
            bmu=bmu,
            bmu_distance=bmu_distance,
            distances=distances,
            memberships=memberships,
            active_units=active_units,
            candidate_diseases=candidate_diseases,
            cumulative_mass=cumulative_mass
        )
    
    def project_batch(
        self, 
        vectors: np.ndarray
    ) -> List[ProjectionResult]:
        """
        Проєктувати batch векторів.
        
        Args:
            vectors: Матриця shape (N, D)
            
        Returns:
            Список ProjectionResult
        """
        results = []
        for vector in vectors:
            results.append(self.project(vector))
        return results
    
    def evaluate_recall(
        self,
        test_vectors: np.ndarray,
        true_diseases: List[str]
    ) -> Dict[str, float]:
        """
        Оцінити Candidate Recall.
        
        Recall = частка випадків коли правильний діагноз
        є серед кандидатів.
        
        Args:
            test_vectors: Тестові вектори shape (N, D)
            true_diseases: Правильні діагнози для кожного вектора
            
        Returns:
            Словник метрик
        """
        hits = 0
        total = len(true_diseases)
        
        candidate_counts = []
        
        for vector, true_disease in zip(test_vectors, true_diseases):
            result = self.project(vector)
            
            candidate_counts.append(len(result.candidate_diseases))
            
            if true_disease in result.candidate_diseases:
                hits += 1
        
        recall = hits / total if total > 0 else 0.0
        
        return {
            "recall": recall,
            "hits": hits,
            "total": total,
            "avg_candidates": np.mean(candidate_counts),
            "min_candidates": np.min(candidate_counts),
            "max_candidates": np.max(candidate_counts),
        }
    
    def __repr__(self) -> str:
        return f"SOMProjector(α={self.config.alpha}, k={self.config.k}, τ={self.config.tau})"
