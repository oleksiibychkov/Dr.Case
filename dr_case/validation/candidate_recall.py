"""
Dr.Case — Метрики якості Candidate Selector

Основна метрика: Candidate Recall
- Частка випадків, де правильний діагноз потрапив до кандидатів
- Ціль: >= 99.5% для production

Додаткові метрики:
- Average Candidates — середня кількість кандидатів
- Candidate Precision — частка правильних серед кандидатів
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from enum import Enum


class RecallLevel(Enum):
    """Рівень recall"""
    PRODUCTION = "production"    # >= 99.5%
    PROTOTYPE = "prototype"      # >= 99.0%
    ACCEPTABLE = "acceptable"    # >= 95.0%
    POOR = "poor"                # < 95%


@dataclass
class CandidateRecallThresholds:
    """Пороги для Candidate Recall"""
    production: float = 0.995
    prototype: float = 0.99
    acceptable: float = 0.95
    
    # Обмеження на кількість кандидатів
    max_candidates_avg: int = 50
    max_candidates_hard: int = 100


@dataclass
class CandidateRecallReport:
    """Звіт про якість Candidate Selector"""
    # Основні метрики
    recall: float                    # Candidate Recall
    recall_level: RecallLevel
    
    # Статистика кандидатів
    avg_candidates: float            # Середня кількість
    min_candidates: int
    max_candidates: int
    std_candidates: float
    
    # Деталі
    total_cases: int
    hits: int                        # Правильний діагноз в кандидатах
    misses: int                      # Правильний діагноз НЕ в кандидатах
    
    # Пропущені діагнози
    missed_diagnoses: List[Tuple[str, List[str]]] = field(default_factory=list)
    # (правильний_діагноз, [кандидати])
    
    # Додаткова статистика
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # P@1, P@5, P@10
    
    def is_production_ready(self) -> bool:
        """Чи готово до production"""
        return self.recall_level == RecallLevel.PRODUCTION
    
    def is_acceptable(self) -> bool:
        """Чи прийнятна якість"""
        return self.recall_level in [
            RecallLevel.PRODUCTION,
            RecallLevel.PROTOTYPE,
            RecallLevel.ACCEPTABLE
        ]
    
    def to_dict(self) -> dict:
        """Серіалізація"""
        return {
            "recall": self.recall,
            "recall_level": self.recall_level.value,
            "avg_candidates": self.avg_candidates,
            "min_candidates": self.min_candidates,
            "max_candidates": self.max_candidates,
            "std_candidates": self.std_candidates,
            "total_cases": self.total_cases,
            "hits": self.hits,
            "misses": self.misses,
            "missed_diagnoses": self.missed_diagnoses[:10],  # Топ-10
            "precision_at_k": self.precision_at_k,
            "is_production_ready": self.is_production_ready(),
            "is_acceptable": self.is_acceptable(),
        }
    
    def __repr__(self) -> str:
        return (
            f"CandidateRecallReport(\n"
            f"  Recall: {self.recall:.4f} ({self.recall_level.value})\n"
            f"  Cases: {self.hits}/{self.total_cases} hits\n"
            f"  Candidates: {self.avg_candidates:.1f} avg ({self.min_candidates}-{self.max_candidates})\n"
            f"  Misses: {self.misses}\n"
            f")"
        )


class CandidateRecallValidator:
    """
    Валідатор якості Candidate Selector.
    
    Приклад використання:
        validator = CandidateRecallValidator()
        
        # Варіант 1: з готовими кандидатами
        report = validator.validate(
            test_cases=[
                ("Influenza", ["Influenza", "Cold", "COVID"]),
                ("Diabetes", ["Hypertension", "Obesity"]),  # miss!
            ]
        )
        
        # Варіант 2: з моделями
        report = validator.validate_with_models(
            som_model=som,
            unit_to_diseases=unit_to_diseases,
            test_data=[(symptom_vector, true_diagnosis), ...]
        )
    """
    
    def __init__(self, thresholds: Optional[CandidateRecallThresholds] = None):
        self.thresholds = thresholds or CandidateRecallThresholds()
    
    def validate(
        self,
        test_cases: List[Tuple[str, List[str]]]
    ) -> CandidateRecallReport:
        """
        Валідація з готовими результатами.
        
        Args:
            test_cases: [(true_diagnosis, candidate_list), ...]
            
        Returns:
            CandidateRecallReport
        """
        if not test_cases:
            return self._empty_report()
        
        hits = 0
        misses = 0
        missed_diagnoses = []
        candidate_counts = []
        
        for true_diag, candidates in test_cases:
            candidate_counts.append(len(candidates))
            
            if true_diag in candidates:
                hits += 1
            else:
                misses += 1
                missed_diagnoses.append((true_diag, candidates[:5]))  # Топ-5 кандидатів
        
        total = len(test_cases)
        recall = hits / total if total > 0 else 0.0
        
        # Статистика кандидатів
        avg_cand = np.mean(candidate_counts)
        std_cand = np.std(candidate_counts)
        min_cand = min(candidate_counts)
        max_cand = max(candidate_counts)
        
        # Рівень recall
        recall_level = self._classify_recall(recall)
        
        # Precision@k
        precision_at_k = self._compute_precision_at_k(test_cases)
        
        return CandidateRecallReport(
            recall=recall,
            recall_level=recall_level,
            avg_candidates=avg_cand,
            min_candidates=min_cand,
            max_candidates=max_cand,
            std_candidates=std_cand,
            total_cases=total,
            hits=hits,
            misses=misses,
            missed_diagnoses=missed_diagnoses,
            precision_at_k=precision_at_k,
        )
    
    def validate_with_models(
        self,
        som_model: Any,
        unit_to_diseases: Dict[int, List[str]],
        test_data: List[Tuple[np.ndarray, str]],
        top_k: int = 10,
        alpha: float = 0.9
    ) -> CandidateRecallReport:
        """
        Валідація з моделями SOM.
        
        Args:
            som_model: MiniSom модель
            unit_to_diseases: {unit_idx: [diseases]}
            test_data: [(symptom_vector, true_diagnosis), ...]
            top_k: Кількість найближчих юнітів
            alpha: Поріг cumulative mass
            
        Returns:
            CandidateRecallReport
        """
        from minisom import MiniSom
        
        if not isinstance(som_model, MiniSom):
            raise ValueError("Очікується MiniSom модель")
        
        test_cases = []
        
        for symptom_vector, true_diagnosis in test_data:
            # Знаходимо кандидатів
            candidates = self._select_candidates(
                som_model, unit_to_diseases, symptom_vector, top_k, alpha
            )
            test_cases.append((true_diagnosis, candidates))
        
        return self.validate(test_cases)
    
    def _select_candidates(
        self,
        som: Any,
        unit_to_diseases: Dict,
        symptom_vector: np.ndarray,
        top_k: int,
        alpha: float
    ) -> List[str]:
        """Вибір кандидатів через SOM"""
        # Обчислюємо відстані до всіх юнітів
        h, w = som._weights.shape[:2]
        distances = []
        
        for i in range(h):
            for j in range(w):
                weight = som._weights[i, j]
                dist = np.linalg.norm(symptom_vector - weight)
                # Зберігаємо як tuple (i, j)
                distances.append(((i, j), dist))
        
        # Сортуємо
        distances.sort(key=lambda x: x[1])
        
        # Топ-k юнітів
        active_units = [d[0] for d in distances[:top_k]]
        
        # Збираємо кандидатів
        # unit_to_diseases може мати ключі як tuple або int
        candidates = set()
        for unit_tuple in active_units:
            # Пробуємо різні формати ключів
            i, j = unit_tuple
            unit_idx = i * w + j
            
            # Варіант 1: tuple
            if unit_tuple in unit_to_diseases:
                candidates.update(unit_to_diseases[unit_tuple])
            # Варіант 2: tuple з np.int64 (конвертуємо)
            elif (int(i), int(j)) in unit_to_diseases:
                candidates.update(unit_to_diseases[(int(i), int(j))])
            # Варіант 3: int індекс
            elif unit_idx in unit_to_diseases:
                candidates.update(unit_to_diseases[unit_idx])
            # Варіант 4: string
            elif str(unit_tuple) in unit_to_diseases:
                candidates.update(unit_to_diseases[str(unit_tuple)])
            elif str(unit_idx) in unit_to_diseases:
                candidates.update(unit_to_diseases[str(unit_idx)])
            # Варіант 5: шукаємо серед ключів (повільно, але надійно)
            else:
                for key in unit_to_diseases.keys():
                    if isinstance(key, tuple) and len(key) == 2:
                        if int(key[0]) == int(i) and int(key[1]) == int(j):
                            candidates.update(unit_to_diseases[key])
                            break
        
        return list(candidates)
    
    def _classify_recall(self, recall: float) -> RecallLevel:
        """Класифікація recall"""
        if recall >= self.thresholds.production:
            return RecallLevel.PRODUCTION
        elif recall >= self.thresholds.prototype:
            return RecallLevel.PROTOTYPE
        elif recall >= self.thresholds.acceptable:
            return RecallLevel.ACCEPTABLE
        else:
            return RecallLevel.POOR
    
    def _compute_precision_at_k(
        self,
        test_cases: List[Tuple[str, List[str]]]
    ) -> Dict[int, float]:
        """Precision@k — частка випадків де правильний діагноз в топ-k"""
        precision = {}
        
        for k in [1, 5, 10]:
            hits = 0
            for true_diag, candidates in test_cases:
                if true_diag in candidates[:k]:
                    hits += 1
            precision[k] = hits / len(test_cases) if test_cases else 0.0
        
        return precision
    
    def _empty_report(self) -> CandidateRecallReport:
        """Порожній звіт"""
        return CandidateRecallReport(
            recall=0.0,
            recall_level=RecallLevel.POOR,
            avg_candidates=0.0,
            min_candidates=0,
            max_candidates=0,
            std_candidates=0.0,
            total_cases=0,
            hits=0,
            misses=0,
        )
    
    def validate_from_checkpoint(
        self,
        som_checkpoint_path: str,
        database_path: str,
        n_samples: int = 1000,
        dropout_rate: float = 0.3
    ) -> CandidateRecallReport:
        """
        Валідація з checkpoint та генерацією тестових даних.
        
        Args:
            som_checkpoint_path: Шлях до SOM checkpoint
            database_path: Шлях до бази хвороб-симптомів
            n_samples: Кількість тестових випадків
            dropout_rate: Частка пропущених симптомів
            
        Returns:
            CandidateRecallReport
        """
        import pickle
        import json
        
        # Завантажуємо SOM
        with open(som_checkpoint_path, 'rb') as f:
            som_data = pickle.load(f)
        
        som_model = som_data.get('som') or som_data.get('model')
        unit_to_diseases = som_data.get('unit_to_diseases', {})
        
        # Завантажуємо базу
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        # Створюємо словник симптомів
        all_symptoms = set()
        for disease_data in database.values():
            all_symptoms.update(disease_data.get('symptoms', []))
        symptom_list = sorted(all_symptoms)
        symptom_to_idx = {s: i for i, s in enumerate(symptom_list)}
        
        # Генеруємо тестові дані
        test_data = []
        diseases = list(database.keys())
        
        np.random.seed(42)
        for _ in range(n_samples):
            # Випадковий діагноз
            disease = np.random.choice(diseases)
            symptoms = database[disease].get('symptoms', [])
            
            if not symptoms:
                continue
            
            # Dropout симптомів
            n_keep = max(1, int(len(symptoms) * (1 - dropout_rate)))
            kept_symptoms = np.random.choice(
                symptoms, size=min(n_keep, len(symptoms)), replace=False
            )
            
            # Створюємо вектор
            vector = np.zeros(len(symptom_list), dtype=np.float32)
            for s in kept_symptoms:
                if s in symptom_to_idx:
                    vector[symptom_to_idx[s]] = 1.0
            
            test_data.append((vector, disease))
        
        return self.validate_with_models(
            som_model=som_model,
            unit_to_diseases=unit_to_diseases,
            test_data=test_data
        )
