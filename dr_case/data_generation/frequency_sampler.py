"""
Dr.Case — Frequency Sampler

Генерація навчальних даних на основі реальних частот симптомів.
Замінює pseudo_generation — використовує реальний розподіл з symptom_frequency.

Логіка:
- Для кожної хвороби є symptom_frequency: {"Fever": 92, "Cough": 87, ...}
- Це означає: з 100 пацієнтів 92 мали Fever, 87 мали Cough
- Генеруємо N "пацієнтів" для кожної хвороби, включаючи симптом з ймовірністю freq/max_freq

Приклад використання:
    sampler = FrequencySampler.from_database("data/unified_disease_symptom_merged.json")
    
    # Згенерувати 100 samples на хворобу
    X, y, labels = sampler.generate(samples_per_disease=100)
    
    # Або з параметрами
    X, y, labels = sampler.generate(
        samples_per_disease=100,
        min_symptoms=2,
        noise_probability=0.05
    )
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SamplerConfig:
    """Конфігурація генерації samples"""
    
    # Кількість samples на хворобу
    samples_per_disease: int = 100
    
    # Мінімальна кількість симптомів у пацієнта
    min_symptoms: int = 2
    
    # Максимальна кількість симптомів (None = без обмеження)
    max_symptoms: Optional[int] = None
    
    # Ймовірність додавання випадкового "шуму" (зайвий симптом)
    noise_probability: float = 0.0
    
    # Ймовірність "dropout" симптому (пацієнт не згадав)
    dropout_probability: float = 0.0
    
    # Random seed для відтворюваності
    random_seed: Optional[int] = 42
    
    # Нормалізувати частоти до [0, 1]
    normalize_frequencies: bool = True
    
    # Використовувати multilabel (пацієнт може мати декілька діагнозів)
    multilabel: bool = False
    
    # Ймовірність коморбідності (другий діагноз)
    comorbidity_probability: float = 0.0


@dataclass
class GeneratedSample:
    """Один згенерований sample"""
    symptoms: List[str]
    diagnosis: str
    symptom_vector: Optional[np.ndarray] = None
    
    # Додаткова інформація
    original_symptoms: List[str] = field(default_factory=list)
    noise_symptoms: List[str] = field(default_factory=list)
    dropped_symptoms: List[str] = field(default_factory=list)


class FrequencySampler:
    """
    Генератор навчальних даних на основі частот симптомів.
    
    Використовує реальний розподіл symptom_frequency для створення
    реалістичних комбінацій симптомів.
    """
    
    def __init__(
        self,
        disease_data: Dict[str, Dict],
        config: Optional[SamplerConfig] = None
    ):
        """
        Args:
            disease_data: Словник {disease_name: {symptoms, symptom_frequency, ...}}
            config: Конфігурація генерації
        """
        self.disease_data = disease_data
        self.config = config or SamplerConfig()
        
        # Встановлюємо seed
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Збираємо всі симптоми
        self._all_symptoms: Set[str] = set()
        for data in disease_data.values():
            self._all_symptoms.update(s.lower() for s in data.get('symptoms', []))
        
        self._all_symptoms_list = sorted(self._all_symptoms)
        self._symptom_to_idx = {s: i for i, s in enumerate(self._all_symptoms_list)}
        
        # Список хвороб
        self._disease_names = list(disease_data.keys())
        self._disease_to_idx = {d: i for i, d in enumerate(self._disease_names)}
    
    @classmethod
    def from_database(
        cls,
        path: str,
        config: Optional[SamplerConfig] = None
    ) -> "FrequencySampler":
        """
        Створити з JSON файлу бази даних.
        
        Args:
            path: Шлях до JSON файлу
            config: Конфігурація
            
        Returns:
            FrequencySampler
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(data, config)
    
    def _get_symptom_probabilities(self, disease: str) -> Dict[str, float]:
        """
        Отримати ймовірності симптомів для хвороби.
        
        Args:
            disease: Назва хвороби
            
        Returns:
            Словник {symptom: probability}
        """
        data = self.disease_data.get(disease, {})
        frequencies = data.get('symptom_frequency', {})
        symptoms = data.get('symptoms', [])
        
        if not frequencies:
            # Якщо немає частот — рівномірний розподіл
            return {s.lower(): 0.7 for s in symptoms}
        
        # Нормалізуємо частоти
        if self.config.normalize_frequencies:
            max_freq = max(frequencies.values()) if frequencies else 1
            probabilities = {
                s.lower(): freq / max_freq 
                for s, freq in frequencies.items()
            }
        else:
            # Припускаємо що частоти вже в діапазоні 0-100
            probabilities = {
                s.lower(): freq / 100.0 
                for s, freq in frequencies.items()
            }
        
        return probabilities
    
    def _sample_symptoms(self, disease: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Згенерувати набір симптомів для одного "пацієнта".
        
        Args:
            disease: Назва хвороби
            
        Returns:
            (final_symptoms, original_symptoms, noise_symptoms)
        """
        probabilities = self._get_symptom_probabilities(disease)
        
        # Генеруємо симптоми за ймовірностями
        original_symptoms = []
        for symptom, prob in probabilities.items():
            if random.random() < prob:
                original_symptoms.append(symptom)
        
        # Dropout — пацієнт "забув" згадати симптом
        dropped_symptoms = []
        if self.config.dropout_probability > 0:
            kept = []
            for s in original_symptoms:
                if random.random() < self.config.dropout_probability:
                    dropped_symptoms.append(s)
                else:
                    kept.append(s)
            original_symptoms = kept
        
        # Забезпечуємо мінімум симптомів
        if len(original_symptoms) < self.config.min_symptoms:
            # Додаємо з найбільш ймовірних
            sorted_symptoms = sorted(probabilities.items(), key=lambda x: -x[1])
            for symptom, _ in sorted_symptoms:
                if symptom not in original_symptoms:
                    original_symptoms.append(symptom)
                    if len(original_symptoms) >= self.config.min_symptoms:
                        break
        
        # Обмежуємо максимум
        if self.config.max_symptoms and len(original_symptoms) > self.config.max_symptoms:
            original_symptoms = random.sample(original_symptoms, self.config.max_symptoms)
        
        # Noise — випадкові зайві симптоми
        noise_symptoms = []
        if self.config.noise_probability > 0:
            other_symptoms = list(self._all_symptoms - set(original_symptoms))
            for symptom in other_symptoms:
                if random.random() < self.config.noise_probability:
                    noise_symptoms.append(symptom)
        
        final_symptoms = original_symptoms + noise_symptoms
        
        return final_symptoms, original_symptoms, noise_symptoms
    
    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Конвертувати список симптомів у бінарний вектор.
        
        Args:
            symptoms: Список симптомів
            
        Returns:
            Бінарний вектор shape (n_symptoms,)
        """
        vector = np.zeros(len(self._all_symptoms_list), dtype=np.float32)
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in self._symptom_to_idx:
                vector[self._symptom_to_idx[symptom_lower]] = 1.0
        
        return vector
    
    def _disease_to_vector(self, diseases: List[str]) -> np.ndarray:
        """
        Конвертувати список діагнозів у multilabel вектор.
        
        Args:
            diseases: Список діагнозів
            
        Returns:
            Бінарний вектор shape (n_diseases,)
        """
        vector = np.zeros(len(self._disease_names), dtype=np.float32)
        
        for disease in diseases:
            if disease in self._disease_to_idx:
                vector[self._disease_to_idx[disease]] = 1.0
        
        return vector
    
    def generate_samples(
        self,
        samples_per_disease: Optional[int] = None,
        diseases: Optional[List[str]] = None
    ) -> List[GeneratedSample]:
        """
        Згенерувати список samples.
        
        Args:
            samples_per_disease: Кількість samples на хворобу (override config)
            diseases: Список хвороб (None = всі)
            
        Returns:
            Список GeneratedSample
        """
        n_samples = samples_per_disease or self.config.samples_per_disease
        target_diseases = diseases or self._disease_names
        
        samples = []
        
        for disease in target_diseases:
            if disease not in self.disease_data:
                continue
            
            for _ in range(n_samples):
                final_symptoms, original, noise = self._sample_symptoms(disease)
                
                sample = GeneratedSample(
                    symptoms=final_symptoms,
                    diagnosis=disease,
                    symptom_vector=self._symptoms_to_vector(final_symptoms),
                    original_symptoms=original,
                    noise_symptoms=noise
                )
                samples.append(sample)
        
        # Перемішуємо
        random.shuffle(samples)
        
        return samples
    
    def generate(
        self,
        samples_per_disease: Optional[int] = None,
        diseases: Optional[List[str]] = None,
        return_vectors: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Згенерувати навчальні дані у форматі numpy arrays.
        
        Args:
            samples_per_disease: Кількість samples на хворобу
            diseases: Список хвороб (None = всі)
            return_vectors: Повертати вектори (True) чи списки (False)
            
        Returns:
            (X, y, disease_labels) де:
            - X: матриця симптомів shape (n_samples, n_symptoms)
            - y: матриця діагнозів shape (n_samples, n_diseases) для multilabel
                 або вектор індексів shape (n_samples,) для single-label
            - disease_labels: список назв діагнозів для кожного sample
        """
        samples = self.generate_samples(samples_per_disease, diseases)
        
        n_samples = len(samples)
        n_symptoms = len(self._all_symptoms_list)
        n_diseases = len(self._disease_names)
        
        # Створюємо матриці
        X = np.zeros((n_samples, n_symptoms), dtype=np.float32)
        
        if self.config.multilabel:
            y = np.zeros((n_samples, n_diseases), dtype=np.float32)
        else:
            y = np.zeros(n_samples, dtype=np.int64)
        
        disease_labels = []
        
        for i, sample in enumerate(samples):
            X[i] = sample.symptom_vector
            
            if self.config.multilabel:
                y[i] = self._disease_to_vector([sample.diagnosis])
            else:
                y[i] = self._disease_to_idx[sample.diagnosis]
            
            disease_labels.append(sample.diagnosis)
        
        return X, y, disease_labels
    
    def generate_train_val_test(
        self,
        samples_per_disease: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Згенерувати train/val/test splits.
        
        Args:
            samples_per_disease: Кількість samples на хворобу
            train_ratio, val_ratio, test_ratio: Пропорції splits
            
        Returns:
            {"train": (X, y), "val": (X, y), "test": (X, y)}
        """
        X, y, _ = self.generate(samples_per_disease)
        
        n = len(X)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            "train": (X[train_idx], y[train_idx]),
            "val": (X[val_idx], y[val_idx]),
            "test": (X[test_idx], y[test_idx])
        }
    
    def get_statistics(self) -> Dict:
        """Отримати статистику даних"""
        symptom_counts = []
        
        for disease, data in self.disease_data.items():
            symptom_counts.append(len(data.get('symptoms', [])))
        
        return {
            "n_diseases": len(self._disease_names),
            "n_symptoms": len(self._all_symptoms_list),
            "avg_symptoms_per_disease": np.mean(symptom_counts),
            "min_symptoms_per_disease": np.min(symptom_counts),
            "max_symptoms_per_disease": np.max(symptom_counts),
            "config": {
                "samples_per_disease": self.config.samples_per_disease,
                "min_symptoms": self.config.min_symptoms,
                "noise_probability": self.config.noise_probability,
                "dropout_probability": self.config.dropout_probability,
            }
        }
    
    @property
    def n_diseases(self) -> int:
        """Кількість хвороб"""
        return len(self._disease_names)
    
    @property
    def n_symptoms(self) -> int:
        """Кількість симптомів"""
        return len(self._all_symptoms_list)
    
    @property
    def disease_names(self) -> List[str]:
        """Список назв хвороб"""
        return self._disease_names.copy()
    
    @property
    def symptom_names(self) -> List[str]:
        """Список назв симптомів"""
        return self._all_symptoms_list.copy()
    
    def __repr__(self) -> str:
        return (
            f"FrequencySampler("
            f"diseases={self.n_diseases}, "
            f"symptoms={self.n_symptoms}, "
            f"samples_per_disease={self.config.samples_per_disease})"
        )
