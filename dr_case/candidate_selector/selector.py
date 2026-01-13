"""
Dr.Case — Candidate Selector

Відбір діагнозів-кандидатів на основі SOM проєкції.
Це ключовий компонент для звуження простору пошуку перед Neural Network.

Pipeline:
1. Пацієнт → вектор симптомів
2. Вектор → SOM проєкція → membership values
3. Membership → активні юніти (α, k, τ)
4. Активні юніти → діагнози-кандидати
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from dr_case.config import CandidateSelectorConfig, get_default_config
from dr_case.schemas import CaseRecord, CandidateDiagnoses, SOMResult
from dr_case.encoding import PatientEncoder, SymptomVocabulary, DiseaseEncoder
from dr_case.som import SOMModel, SOMProjector, ProjectionResult


@dataclass
class SelectionResult:
    """
    Результат відбору кандидатів.
    
    Містить всю інформацію для подальшої обробки:
    - Список кандидатів
    - SOM проєкція
    - Вектор симптомів
    """
    # Кандидати
    candidates: List[str]
    candidate_count: int
    
    # SOM інформація
    bmu: Tuple[int, int]
    bmu_distance: float
    active_units: List[Tuple[int, int]]
    memberships: Dict[Tuple[int, int], float]
    cumulative_mass: float
    
    # Вхідні дані
    present_symptoms: List[str]
    absent_symptoms: List[str]
    symptom_vector: np.ndarray
    
    def to_candidate_diagnoses(self) -> CandidateDiagnoses:
        """Конвертувати в CandidateDiagnoses schema"""
        unit_memberships = {
            f"{u[0]}_{u[1]}": m 
            for u, m in self.memberships.items()
        }
        
        return CandidateDiagnoses(
            candidates=self.candidates,
            unit_memberships=unit_memberships,
            bmu_id=f"{self.bmu[0]}_{self.bmu[1]}",
            bmu_distance=self.bmu_distance
        )
    
    def to_som_result(self) -> SOMResult:
        """Конвертувати в SOMResult schema"""
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
    
    def contains(self, disease_name: str) -> bool:
        """Перевірити чи діагноз є серед кандидатів"""
        return disease_name in self.candidates
    
    def get_top_candidates(self, n: int = 10) -> List[str]:
        """Отримати топ-N кандидатів"""
        return self.candidates[:n]


class CandidateSelector:
    """
    Відбір діагнозів-кандидатів на основі SOM.
    
    Приклад використання:
        # Ініціалізація
        selector = CandidateSelector.from_model_file("models/som_optimized.pkl")
        
        # Відбір за списком симптомів
        result = selector.select(["fever", "headache", "cough"])
        print(f"Candidates: {result.candidates}")
        
        # Відбір за CaseRecord
        case = CaseRecord(symptoms=[...])
        result = selector.select_from_case(case)
        
        # Відбір за вектором
        result = selector.select_from_vector(patient_vector)
    """
    
    def __init__(
        self,
        som_model: SOMModel,
        vocabulary: SymptomVocabulary,
        config: Optional[CandidateSelectorConfig] = None
    ):
        """
        Args:
            som_model: Навчена модель SOM
            vocabulary: Словник симптомів
            config: Конфігурація Candidate Selector
        """
        self.som_model = som_model
        self.vocabulary = vocabulary
        self.config = config or CandidateSelectorConfig()
        
        # Створюємо компоненти
        self.patient_encoder = PatientEncoder(vocabulary)
        self.projector = SOMProjector(som_model, self.config)
    
   
    @classmethod
    def from_model_file(
        cls,
        model_path: str,
        database_path: Optional[str] = None,
        config: Optional[CandidateSelectorConfig] = None,
        tuning_result_path: Optional[str] = None
    ) -> "CandidateSelector":
        """
        Створити з файлу моделі.
        
        Args:
            model_path: Шлях до .pkl файлу SOM моделі
            database_path: Шлях до бази даних (для vocabulary)
            config: Конфігурація (якщо None - спробує з tuning_result)
            tuning_result_path: Шлях до JSON з результатами оптимізації
            
        Returns:
            CandidateSelector
        """
        import json
        
        # Завантажуємо SOM
        som_model = SOMModel.load(model_path)
        
        # Створюємо vocabulary
        model_dir = Path(model_path).parent
        
        if database_path is None:
            default_db = model_dir.parent / "data" / "unified_disease_symptom_data_full.json"
            if default_db.exists():
                database_path = str(default_db)
            else:
                raise FileNotFoundError(f"Database not found at {default_db}")
        
        vocabulary = SymptomVocabulary.from_database(database_path)
        
        # Завантажуємо конфігурацію
        if config is None:
            # Спробуємо знайти tuning_result.json
            if tuning_result_path is None:
                tuning_result_path = model_dir / "som_tuning_result.json"
            
            if Path(tuning_result_path).exists():
                with open(tuning_result_path, "r") as f:
                    tuning_data = json.load(f)
                
                params = tuning_data.get("best_params", {})
                config = CandidateSelectorConfig(
                    alpha=params.get("alpha", 0.9),
                    k=params.get("k", 6),
                    tau=params.get("tau", 0.01),
                    membership_lambda=params.get("lambda_param", 1.0)
                )
                print(f"Loaded config from {tuning_result_path}")
            else:
                config = CandidateSelectorConfig()
        
        return cls(som_model, vocabulary, config)    
    @classmethod
    def from_database(
        cls,
        database_path: str,
        som_config: Optional[dict] = None,
        selector_config: Optional[CandidateSelectorConfig] = None,
        epochs: int = 500
    ) -> "CandidateSelector":
        """
        Створити та навчити SOM з бази даних.
        
        Args:
            database_path: Шлях до JSON бази даних
            som_config: Параметри SOM (grid_size, etc.)
            selector_config: Конфігурація Candidate Selector
            epochs: Кількість епох навчання
            
        Returns:
            CandidateSelector з навченою SOM
        """
        from dr_case.som import SOMTrainer
        from dr_case.config import SOMConfig
        
        # Конфігурація SOM
        if som_config:
            config = SOMConfig(**som_config)
        else:
            config = SOMConfig(
                grid_height=30,
                grid_width=30,
                epochs=epochs
            )
        
        # Навчаємо SOM
        trainer = SOMTrainer(config)
        som_model, metrics = trainer.train_from_database(database_path)
        
        print(f"SOM trained: QE={metrics['qe']:.4f}, TE={metrics['te']:.4f}")
        
        # Vocabulary
        vocabulary = SymptomVocabulary.from_database(database_path)
        
        return cls(som_model, vocabulary, selector_config)
    
    def select(
        self,
        present_symptoms: List[str],
        absent_symptoms: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None
    ) -> SelectionResult:
        """
        Відібрати кандидатів за списком симптомів.
        
        Args:
            present_symptoms: Присутні симптоми
            absent_symptoms: Відсутні симптоми
            alpha, k, tau: Override параметрів
            
        Returns:
            SelectionResult з кандидатами
        """
        absent_symptoms = absent_symptoms or []
        
        # Кодуємо симптоми
        if absent_symptoms:
            symptom_vector = self.patient_encoder.encode_with_negatives(
                present=present_symptoms,
                absent=absent_symptoms
            )
        else:
            symptom_vector = self.patient_encoder.encode(present_symptoms)
        
        # Проєктуємо
        projection = self.projector.project(
            symptom_vector,
            alpha=alpha,
            k=k,
            tau=tau
        )
        
        return SelectionResult(
            candidates=projection.candidate_diseases,
            candidate_count=len(projection.candidate_diseases),
            bmu=projection.bmu,
            bmu_distance=projection.bmu_distance,
            active_units=projection.active_units,
            memberships=projection.memberships,
            cumulative_mass=projection.cumulative_mass,
            present_symptoms=present_symptoms,
            absent_symptoms=absent_symptoms,
            symptom_vector=symptom_vector
        )
    
    def select_from_case(
        self,
        case: CaseRecord,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None
    ) -> SelectionResult:
        """
        Відібрати кандидатів з CaseRecord.
        
        Args:
            case: Клінічний випадок
            alpha, k, tau: Override параметрів
            
        Returns:
            SelectionResult
        """
        return self.select(
            present_symptoms=case.present_symptom_names,
            absent_symptoms=case.absent_symptom_names,
            alpha=alpha,
            k=k,
            tau=tau
        )
    
    def select_from_vector(
        self,
        symptom_vector: np.ndarray,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None
    ) -> SelectionResult:
        """
        Відібрати кандидатів за вектором симптомів.
        
        Args:
            symptom_vector: Вектор симптомів shape (D,)
            alpha, k, tau: Override параметрів
            
        Returns:
            SelectionResult
        """
        # Проєктуємо
        projection = self.projector.project(
            symptom_vector,
            alpha=alpha,
            k=k,
            tau=tau
        )
        
        # Декодуємо вектор назад в симптоми
        present, absent = self.patient_encoder.decode(symptom_vector)
        
        return SelectionResult(
            candidates=projection.candidate_diseases,
            candidate_count=len(projection.candidate_diseases),
            bmu=projection.bmu,
            bmu_distance=projection.bmu_distance,
            active_units=projection.active_units,
            memberships=projection.memberships,
            cumulative_mass=projection.cumulative_mass,
            present_symptoms=present,
            absent_symptoms=absent,
            symptom_vector=symptom_vector
        )
    
    def select_batch(
        self,
        symptom_lists: List[List[str]]
    ) -> List[SelectionResult]:
        """
        Відібрати кандидатів для batch симптомів.
        
        Args:
            symptom_lists: Список списків симптомів
            
        Returns:
            Список SelectionResult
        """
        results = []
        for symptoms in symptom_lists:
            results.append(self.select(symptoms))
        return results
    
    def evaluate_recall(
        self,
        test_symptoms: List[List[str]],
        true_diseases: List[str]
    ) -> Dict[str, float]:
        """
        Оцінити Candidate Recall.
        
        Args:
            test_symptoms: Списки симптомів для кожного тесту
            true_diseases: Правильні діагнози
            
        Returns:
            Метрики recall
        """
        hits = 0
        candidate_counts = []
        
        for symptoms, true_disease in zip(test_symptoms, true_diseases):
            result = self.select(symptoms)
            candidate_counts.append(result.candidate_count)
            
            if result.contains(true_disease):
                hits += 1
        
        total = len(true_diseases)
        
        return {
            "recall": hits / total if total > 0 else 0.0,
            "hits": hits,
            "total": total,
            "avg_candidates": np.mean(candidate_counts),
            "min_candidates": np.min(candidate_counts),
            "max_candidates": np.max(candidate_counts),
        }
    
    def get_diseases_in_neighborhood(
        self,
        present_symptoms: List[str],
        radius: int = 3
    ) -> Dict[str, List[str]]:
        """
        Отримати діагнози в сусідстві BMU.
        
        Корисно для аналізу та пояснень.
        
        Args:
            present_symptoms: Симптоми пацієнта
            radius: Радіус сусідства
            
        Returns:
            Словник {unit_id: [diseases]}
        """
        # Кодуємо та знаходимо BMU
        vector = self.patient_encoder.encode(present_symptoms)
        bmu = self.som_model.get_bmu(vector)
        
        # Збираємо діагнози з сусідства
        result = {}
        
        # BMU
        bmu_diseases = self.som_model.get_diseases_in_unit(bmu)
        if bmu_diseases:
            result[f"{bmu[0]}_{bmu[1]}"] = bmu_diseases
        
        # Сусіди
        neighbors = self.som_model.get_neighbors(bmu, radius)
        for unit in neighbors:
            diseases = self.som_model.get_diseases_in_unit(unit)
            if diseases:
                result[f"{unit[0]}_{unit[1]}"] = diseases
        
        return result
    
    def update_config(
        self,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None,
        membership_lambda: Optional[float] = None
    ) -> None:
        """
        Оновити параметри конфігурації.
        
        Args:
            alpha, k, tau, membership_lambda: Нові значення
        """
        if alpha is not None:
            self.config.alpha = alpha
        if k is not None:
            self.config.k = k
        if tau is not None:
            self.config.tau = tau
        if membership_lambda is not None:
            self.config.membership_lambda = membership_lambda
        
        # Оновлюємо projector
        self.projector = SOMProjector(self.som_model, self.config)
    
    def save_model(self, path: str) -> None:
        """Зберегти SOM модель"""
        self.som_model.save(path)
    
    @property
    def n_symptoms(self) -> int:
        """Кількість симптомів"""
        return self.vocabulary.size
    
    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Розмір карти SOM"""
        return self.som_model.grid_shape
    
    def __repr__(self) -> str:
        return (
            f"CandidateSelector("
            f"grid={self.grid_shape[0]}x{self.grid_shape[1]}, "
            f"α={self.config.alpha}, k={self.config.k}, τ={self.config.tau})"
        )
