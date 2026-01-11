"""
Dr.Case — Кодування діагнозів

Перетворює діагнози у бінарні вектори симптомів.
Кожен діагноз представлений вектором x ∈ {0,1}^D, де D — кількість симптомів.
"""

import numpy as np
from typing import Dict, List, Optional
from .data_loader import DiseaseDatabaseLoader
from .symptom_vocabulary import SymptomVocabulary


class DiseaseEncoder:
    """
    Кодувальник діагнозів у вектори.
    
    Кожен діагноз кодується як бінарний вектор, де:
    - 1 = симптом присутній для цього діагнозу
    - 0 = симптом відсутній
    
    Приклад використання:
        encoder = DiseaseEncoder.from_database("data/unified_disease_symptom_data_full.json")
        
        # Один діагноз → вектор
        vector = encoder.encode("Influenza")  # np.array shape (461,)
        
        # Всі діагнози → матриця
        matrix = encoder.encode_all()  # np.array shape (842, 461)
    """
    
    def __init__(self, vocabulary: SymptomVocabulary, loader: DiseaseDatabaseLoader):
        """
        Args:
            vocabulary: Словник симптомів
            loader: Завантажувач бази даних
        """
        self.vocabulary = vocabulary
        self.loader = loader
        
        # Кеш векторів
        self._cache: Dict[str, np.ndarray] = {}
        
        # Словник disease_name → index
        self._disease_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(sorted(loader.disease_names))
        }
        self._idx_to_disease: Dict[int, str] = {
            idx: name for name, idx in self._disease_to_idx.items()
        }
    
    @classmethod
    def from_database(cls, path: Optional[str] = None) -> "DiseaseEncoder":
        """
        Створити енкодер з бази даних.
        
        Args:
            path: Шлях до JSON файлу
            
        Returns:
            DiseaseEncoder
        """
        loader = DiseaseDatabaseLoader(path)
        vocabulary = SymptomVocabulary.from_database(path)
        return cls(vocabulary, loader)
    
    @property
    def vector_dim(self) -> int:
        """Розмірність вектора (кількість симптомів)"""
        return self.vocabulary.size
    
    @property
    def disease_count(self) -> int:
        """Кількість діагнозів"""
        return self.loader.disease_count
    
    @property
    def disease_names(self) -> List[str]:
        """Список назв діагнозів (в порядку індексів)"""
        return [self._idx_to_disease[i] for i in range(self.disease_count)]
    
    def disease_to_index(self, disease_name: str) -> Optional[int]:
        """Отримати індекс діагнозу"""
        return self._disease_to_idx.get(disease_name)
    
    def index_to_disease(self, index: int) -> Optional[str]:
        """Отримати назву діагнозу за індексом"""
        return self._idx_to_disease.get(index)
    
    def encode(self, disease_name: str, normalize: bool = False) -> Optional[np.ndarray]:
        """
        Закодувати діагноз у вектор.
        
        Args:
            disease_name: Назва діагнозу
            normalize: Чи нормалізувати вектор (L2)
            
        Returns:
            Бінарний вектор shape (D,) або None якщо діагноз не знайдено
        """
        # Перевіряємо кеш
        cache_key = f"{disease_name}_{normalize}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Отримуємо симптоми діагнозу
        symptoms = self.loader.get_symptoms(disease_name)
        if not symptoms:
            return None
        
        # Створюємо вектор
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        
        for symptom in symptoms:
            idx = self.vocabulary.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = 1.0
        
        # Нормалізація
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        # Кешуємо
        self._cache[cache_key] = vector
        
        return vector.copy()
    
    def encode_all(self, normalize: bool = False) -> np.ndarray:
        """
        Закодувати всі діагнози у матрицю.
        
        Args:
            normalize: Чи нормалізувати вектори
            
        Returns:
            Матриця shape (N_diseases, D) де кожен рядок — вектор діагнозу
        """
        matrix = np.zeros((self.disease_count, self.vector_dim), dtype=np.float32)
        
        for disease_name in self.disease_names:
            idx = self.disease_to_index(disease_name)
            vector = self.encode(disease_name, normalize=normalize)
            if vector is not None:
                matrix[idx] = vector
        
        return matrix
    
    def decode(self, vector: np.ndarray, threshold: float = 0.5) -> List[str]:
        """
        Декодувати вектор назад у симптоми.
        
        Args:
            vector: Вектор симптомів
            threshold: Поріг для бінаризації
            
        Returns:
            Список симптомів
        """
        symptoms = []
        for idx in np.where(vector >= threshold)[0]:
            symptom = self.vocabulary.index_to_symptom(int(idx))
            if symptom:
                symptoms.append(symptom)
        return symptoms
    
    def get_symptom_frequencies(self) -> Dict[str, int]:
        """
        Отримати частоту кожного симптому (в скількох діагнозах зустрічається).
        
        Returns:
            Словник {symptom: count}
        """
        frequencies = {}
        
        for symptom in self.vocabulary.symptoms:
            count = len(self.loader.get_diseases_by_symptom(symptom))
            frequencies[symptom] = count
        
        return frequencies
    
    def get_disease_similarity(self, disease1: str, disease2: str) -> float:
        """
        Обчислити схожість двох діагнозів (Jaccard similarity).
        
        Args:
            disease1: Перший діагноз
            disease2: Другий діагноз
            
        Returns:
            Коефіцієнт Jaccard [0, 1]
        """
        vec1 = self.encode(disease1)
        vec2 = self.encode(disease2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        intersection = np.sum(np.minimum(vec1, vec2))
        union = np.sum(np.maximum(vec1, vec2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def clear_cache(self) -> None:
        """Очистити кеш векторів"""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"DiseaseEncoder(diseases={self.disease_count}, symptoms={self.vector_dim})"
