"""
Dr.Case — Кодування симптомів пацієнта

Перетворює список симптомів пацієнта у вектор для подачі в SOM та NN.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .symptom_vocabulary import SymptomVocabulary


class PatientEncoder:
    """
    Кодувальник симптомів пацієнта.
    
    Підтримує:
    - Бінарне кодування (symptom present = 1)
    - Вагове кодування (symptom present = severity)
    - Негативні симптоми (symptom explicitly absent = -1 або 0)
    
    Приклад використання:
        encoder = PatientEncoder(vocabulary)
        
        # Прості симптоми
        vector = encoder.encode(["fever", "headache", "cough"])
        
        # З вагами (severity)
        vector = encoder.encode_weighted({
            "fever": 0.8,
            "headache": 0.6,
            "cough": 0.4
        })
        
        # З негативними симптомами
        vector = encoder.encode_with_negatives(
            present=["fever", "cough"],
            absent=["rash", "vomiting"]
        )
    """
    
    def __init__(self, vocabulary: SymptomVocabulary):
        """
        Args:
            vocabulary: Словник симптомів
        """
        self.vocabulary = vocabulary
    
    @property
    def vector_dim(self) -> int:
        """Розмірність вектора"""
        return self.vocabulary.size
    
    def encode(
        self, 
        symptoms: List[str], 
        normalize: bool = False
    ) -> np.ndarray:
        """
        Закодувати список симптомів у бінарний вектор.
        
        Args:
            symptoms: Список симптомів пацієнта
            normalize: Чи нормалізувати вектор (L2)
            
        Returns:
            Вектор shape (D,)
        """
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        
        for symptom in symptoms:
            idx = self.vocabulary.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = 1.0
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def encode_weighted(
        self, 
        symptoms: Dict[str, float], 
        normalize: bool = False
    ) -> np.ndarray:
        """
        Закодувати симптоми з вагами (severity).
        
        Args:
            symptoms: Словник {symptom: severity}, severity ∈ [0, 1]
            normalize: Чи нормалізувати вектор
            
        Returns:
            Вектор shape (D,)
        """
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        
        for symptom, weight in symptoms.items():
            idx = self.vocabulary.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = float(weight)
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def encode_with_negatives(
        self,
        present: List[str],
        absent: List[str],
        normalize: bool = False,
        absent_value: float = -1.0
    ) -> np.ndarray:
        """
        Закодувати симптоми з урахуванням явно відсутніх.
        
        Це важливо для диференціальної діагностики:
        - Присутні симптоми = 1
        - Явно відсутні = absent_value (зазвичай -1 або 0)
        - Невідомі = 0
        
        Args:
            present: Симптоми, що присутні
            absent: Симптоми, що явно відсутні
            normalize: Чи нормалізувати вектор
            absent_value: Значення для відсутніх симптомів
            
        Returns:
            Вектор shape (D,)
        """
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        
        # Присутні
        for symptom in present:
            idx = self.vocabulary.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = 1.0
        
        # Явно відсутні
        for symptom in absent:
            idx = self.vocabulary.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = absent_value
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def encode_iterative(
        self,
        initial_symptoms: List[str],
        additional_symptoms: List[str],
        negated_symptoms: List[str],
        normalize: bool = False
    ) -> np.ndarray:
        """
        Кодування для ітеративного процесу діагностики.
        
        Поєднує початкові скарги, додаткові симптоми (з уточнюючих питань)
        та заперечені симптоми.
        
        Args:
            initial_symptoms: Початкові симптоми (скарги пацієнта)
            additional_symptoms: Додаткові симптоми (виявлені при опитуванні)
            negated_symptoms: Симптоми, що пацієнт заперечив
            normalize: Чи нормалізувати вектор
            
        Returns:
            Вектор shape (D,)
        """
        all_present = list(set(initial_symptoms + additional_symptoms))
        return self.encode_with_negatives(
            present=all_present,
            absent=negated_symptoms,
            normalize=normalize
        )
    
    def decode(
        self, 
        vector: np.ndarray, 
        threshold: float = 0.5
    ) -> Tuple[List[str], List[str]]:
        """
        Декодувати вектор назад у симптоми.
        
        Args:
            vector: Вектор симптомів
            threshold: Поріг для класифікації присутності
            
        Returns:
            Tuple (present_symptoms, absent_symptoms)
        """
        present = []
        absent = []
        
        for idx in range(len(vector)):
            symptom = self.vocabulary.index_to_symptom(idx)
            if symptom:
                if vector[idx] >= threshold:
                    present.append(symptom)
                elif vector[idx] < 0:
                    absent.append(symptom)
        
        return present, absent
    
    def get_unknown_symptoms(
        self,
        present: List[str],
        absent: List[str]
    ) -> List[str]:
        """
        Отримати симптоми, про які немає інформації.
        
        Корисно для вибору наступного питання.
        
        Args:
            present: Присутні симптоми
            absent: Відсутні симптоми
            
        Returns:
            Список невідомих симптомів
        """
        known = set(s.strip().lower() for s in present + absent)
        unknown = []
        
        for symptom in self.vocabulary.symptoms:
            if symptom not in known:
                unknown.append(symptom)
        
        return unknown
    
    def symptom_coverage(self, symptoms: List[str]) -> float:
        """
        Обчислити покриття симптомів (яка частка словника відома).
        
        Args:
            symptoms: Відомі симптоми (присутні + відсутні)
            
        Returns:
            Частка покриття [0, 1]
        """
        known_count = 0
        for symptom in symptoms:
            if self.vocabulary.has_symptom(symptom):
                known_count += 1
        
        return known_count / self.vocabulary.size if self.vocabulary.size > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"PatientEncoder(vector_dim={self.vector_dim})"
