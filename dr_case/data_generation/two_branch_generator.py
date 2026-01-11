"""
Dr.Case — Генератор даних для двогілкової архітектури

Генерує трійки (symptom_vector, som_context, disease_label):
1. symptom_vector — з FrequencySampler (на основі реальних частот)
2. som_context — membership до top-k SOM юнітів
3. disease_label — індекс діагнозу

ВАЖЛИВО:
- symptom_vector буде L2-нормалізовано в моделі
- som_context вже нормалізовано (softmax в SOM projector)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from dr_case.data_generation import FrequencySampler, SamplerConfig
from dr_case.som import SOMModel
from dr_case.som.projector import SOMProjector
from dr_case.config import CandidateSelectorConfig


@dataclass
class TwoBranchSamplerConfig:
    """Конфігурація генератора для двогілкової архітектури"""
    
    # FrequencySampler config
    samples_per_disease: int = 100
    min_symptoms: int = 2
    noise_probability: float = 0.02
    dropout_probability: float = 0.15
    
    # SOM context config
    som_k: int = 10  # Кількість top юнітів для context
    som_alpha: float = 0.95
    som_tau: float = 0.001
    
    # Random seed
    random_seed: int = 42


class TwoBranchDataGenerator:
    """
    Генератор даних для двогілкової NN.
    
    Приклад використання:
        generator = TwoBranchDataGenerator.from_files(
            database_path="data/unified_disease_symptom_merged.json",
            som_path="models/som_merged.pkl"
        )
        
        X_symptoms, X_som, y, labels = generator.generate(samples_per_disease=100)
        
        splits = generator.generate_train_val_test()
    """
    
    def __init__(
        self,
        sampler: FrequencySampler,
        som_model: SOMModel,
        projector: SOMProjector,
        config: Optional[TwoBranchSamplerConfig] = None
    ):
        """
        Args:
            sampler: FrequencySampler для генерації симптомів
            som_model: Навчена SOM модель
            projector: SOM Projector для обчислення membership
            config: Конфігурація
        """
        self.sampler = sampler
        self.som = som_model
        self.projector = projector
        self.config = config or TwoBranchSamplerConfig()
        
        # Мапінг disease_name → index
        self.disease_names = sampler.disease_names
        self.disease_to_idx = {name: i for i, name in enumerate(self.disease_names)}
        
        # Симптоми
        self.symptom_names = sampler.symptom_names
        self.n_symptoms = len(self.symptom_names)
        self.n_diseases = len(self.disease_names)
        self.som_dim = self.config.som_k
    
    @classmethod
    def from_files(
        cls,
        database_path: str,
        som_path: str,
        config: Optional[TwoBranchSamplerConfig] = None
    ) -> "TwoBranchDataGenerator":
        """
        Створити генератор з файлів.
        
        Args:
            database_path: Шлях до JSON бази даних
            som_path: Шлях до збереженої SOM моделі
            config: Конфігурація
            
        Returns:
            TwoBranchDataGenerator
        """
        config = config or TwoBranchSamplerConfig()
        
        # FrequencySampler
        sampler_config = SamplerConfig(
            samples_per_disease=config.samples_per_disease,
            min_symptoms=config.min_symptoms,
            noise_probability=config.noise_probability,
            dropout_probability=config.dropout_probability,
            random_seed=config.random_seed,
        )
        sampler = FrequencySampler.from_database(database_path, sampler_config)
        
        # SOM
        som_model = SOMModel.load(som_path)
        
        # Projector
        selector_config = CandidateSelectorConfig(
            alpha=config.som_alpha,
            k=config.som_k,
            tau=config.som_tau,
        )
        projector = SOMProjector(som_model, selector_config)
        
        return cls(sampler, som_model, projector, config)
    
    def _get_som_context(self, symptom_vector: np.ndarray) -> np.ndarray:
        """
        Отримати SOM context для вектора симптомів.
        
        Args:
            symptom_vector: Бінарний вектор симптомів (D,)
            
        Returns:
            SOM context — membership для top-k юнітів (k,)
        """
        # Проєкція на SOM
        result = self.projector.project(symptom_vector)
        
        # Отримуємо top-k memberships
        # Сортуємо за membership і беремо top-k
        sorted_memberships = sorted(
            result.memberships.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Формуємо вектор context
        som_context = np.zeros(self.som_dim, dtype=np.float32)
        
        for i, (unit, membership) in enumerate(sorted_memberships[:self.som_dim]):
            som_context[i] = membership
        
        # Ренормалізуємо щоб сума = 1 (якщо взяли менше k юнітів)
        total = som_context.sum()
        if total > 0:
            som_context = som_context / total
        
        return som_context
    
    def generate(
        self,
        samples_per_disease: Optional[int] = None,
        diseases: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Згенерувати дані для навчання.
        
        Args:
            samples_per_disease: Кількість samples на хворобу
            diseases: Список хвороб (None = всі)
            verbose: Виводити прогрес
            
        Returns:
            (X_symptoms, X_som, y, labels):
            - X_symptoms: shape (N, n_symptoms) — вектори симптомів
            - X_som: shape (N, som_dim) — SOM context
            - y: shape (N,) — індекси діагнозів
            - labels: список назв діагнозів
        """
        n_samples = samples_per_disease or self.config.samples_per_disease
        target_diseases = diseases or self.disease_names
        
        # Генеруємо через FrequencySampler
        samples = self.sampler.generate_samples(n_samples, target_diseases)
        
        total = len(samples)
        
        if verbose:
            print(f"Generating SOM context for {total} samples...")
        
        # Алокуємо масиви
        X_symptoms = np.zeros((total, self.n_symptoms), dtype=np.float32)
        X_som = np.zeros((total, self.som_dim), dtype=np.float32)
        y = np.zeros(total, dtype=np.int64)
        labels = []
        
        # Заповнюємо
        for i, sample in enumerate(samples):
            # Symptom vector
            X_symptoms[i] = sample.symptom_vector
            
            # SOM context
            X_som[i] = self._get_som_context(sample.symptom_vector)
            
            # Label
            y[i] = self.disease_to_idx[sample.diagnosis]
            labels.append(sample.diagnosis)
            
            # Progress
            if verbose and (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{total} samples...")
        
        if verbose:
            print(f"  Done! Generated {total} samples.")
        
        return X_symptoms, X_som, y, labels
    
    def generate_train_val_test(
        self,
        samples_per_disease: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        verbose: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Згенерувати train/val/test splits.
        
        Args:
            samples_per_disease: Кількість samples на хворобу
            train_ratio, val_ratio, test_ratio: Пропорції
            verbose: Виводити прогрес
            
        Returns:
            {"train": (X_symptoms, X_som, y), "val": ..., "test": ...}
        """
        X_symptoms, X_som, y, _ = self.generate(samples_per_disease, verbose=verbose)
        
        n = len(y)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            "train": (X_symptoms[train_idx], X_som[train_idx], y[train_idx]),
            "val": (X_symptoms[val_idx], X_som[val_idx], y[val_idx]),
            "test": (X_symptoms[test_idx], X_som[test_idx], y[test_idx]),
        }
    
    def get_dimensions(self) -> Dict[str, int]:
        """Отримати розмірності для моделі"""
        return {
            "n_symptoms": self.n_symptoms,
            "n_diseases": self.n_diseases,
            "som_dim": self.som_dim,
        }
    
    def __repr__(self) -> str:
        return (
            f"TwoBranchDataGenerator(\n"
            f"  diseases={self.n_diseases}, symptoms={self.n_symptoms},\n"
            f"  som_dim={self.som_dim}, samples_per_disease={self.config.samples_per_disease}\n"
            f")"
        )
