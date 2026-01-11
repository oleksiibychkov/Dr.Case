"""
Dr.Case — Модуль кодування (encoding)

Векторизація симптомів та діагнозів для подачі в SOM та Neural Network.

Компоненти:
- DiseaseDatabaseLoader: Завантаження бази даних з JSON
- SymptomVocabulary: Словник симптомів (symptom ↔ index)
- DiseaseEncoder: Кодування діагнозів у вектори
- PatientEncoder: Кодування симптомів пацієнта

Приклад використання:
    from dr_case.encoding import (
        DiseaseDatabaseLoader,
        SymptomVocabulary,
        DiseaseEncoder,
        PatientEncoder
    )
    
    # Завантаження бази
    loader = DiseaseDatabaseLoader("data/unified_disease_symptom_data_full.json")
    print(f"Діагнозів: {loader.disease_count}")
    print(f"Симптомів: {loader.symptom_count}")
    
    # Створення енкодерів
    encoder = DiseaseEncoder.from_database()
    
    # Кодування діагнозу
    flu_vector = encoder.encode("Influenza")
    
    # Кодування всіх діагнозів для SOM
    disease_matrix = encoder.encode_all()  # shape (842, 461)
    
    # Кодування пацієнта
    patient_encoder = PatientEncoder(encoder.vocabulary)
    patient_vector = patient_encoder.encode(["fever", "headache", "cough"])
"""

from .data_loader import DiseaseDatabaseLoader, DiseaseRecord
from .symptom_vocabulary import SymptomVocabulary
from .disease_encoder import DiseaseEncoder
from .patient_encoder import PatientEncoder


__all__ = [
    "DiseaseDatabaseLoader",
    "DiseaseRecord",
    "SymptomVocabulary",
    "DiseaseEncoder",
    "PatientEncoder",
]
