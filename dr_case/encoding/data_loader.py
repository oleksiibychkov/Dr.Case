"""
Dr.Case — Завантаження бази даних

Завантажує unified_disease_symptom_data_full.json та надає
зручний доступ до діагнозів і симптомів.

Структура JSON:
{
  "Disease Name": {
    "disease": { "doid_id": ..., "doid_name": ... },
    "symptoms": ["symptom1", "symptom2", ...],
    "sources": [...],
    "symptom_frequency": { "symptom1": count, ... }
  },
  ...
}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class DiseaseRecord:
    """Запис про діагноз"""
    name: str
    symptoms: List[str]
    symptom_frequency: Dict[str, int]
    sources: List[str]
    
    @property
    def symptom_count(self) -> int:
        return len(self.symptoms)


class DiseaseDatabaseLoader:
    """
    Завантажувач бази даних діагнозів.
    
    Приклад використання:
        loader = DiseaseDatabaseLoader("data/unified_disease_symptom_data_full.json")
        
        print(f"Діагнозів: {loader.disease_count}")
        print(f"Симптомів: {loader.symptom_count}")
        
        # Отримати всі симптоми діагнозу
        symptoms = loader.get_symptoms("Influenza")
        
        # Отримати всі діагнози з симптомом
        diseases = loader.get_diseases_by_symptom("fever")
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Args:
            path: Шлях до JSON файлу. Якщо None, використовує шлях за замовчуванням.
        """
        if path is None:
            # Шлях відносно кореня проекту
            path = Path(__file__).parent.parent.parent / "data" / "unified_disease_symptom_data_full.json"
        
        self.path = Path(path)
        self._raw_data: Dict = {}
        self._diseases: Dict[str, DiseaseRecord] = {}
        self._all_symptoms: Set[str] = set()
        self._symptom_to_diseases: Dict[str, List[str]] = {}
        
        self._load()
    
    def _load(self) -> None:
        """Завантажити та проіндексувати дані"""
        if not self.path.exists():
            raise FileNotFoundError(f"Database file not found: {self.path}")
        
        with open(self.path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)
        
        # Індексуємо дані
        # Структура: { "Disease Name": { "symptoms": [...], ... }, ... }
        for disease_name, disease_data in self._raw_data.items():
            # Отримуємо симптоми
            symptoms_raw = disease_data.get("symptoms", [])
            
            # Нормалізуємо симптоми (lowercase, strip)
            symptoms = [s.strip().lower() for s in symptoms_raw if s.strip()]
            
            # Отримуємо частоти симптомів
            symptom_frequency = disease_data.get("symptom_frequency", {})
            # Нормалізуємо ключі
            symptom_frequency = {k.strip().lower(): v for k, v in symptom_frequency.items()}
            
            # Отримуємо джерела
            sources = disease_data.get("sources", [])
            
            # Зберігаємо
            disease_record = DiseaseRecord(
                name=disease_name,
                symptoms=symptoms,
                symptom_frequency=symptom_frequency,
                sources=sources
            )
            self._diseases[disease_name] = disease_record
            
            # Оновлюємо множину всіх симптомів
            self._all_symptoms.update(symptoms)
            
            # Індекс symptom → diseases
            for symptom in symptoms:
                if symptom not in self._symptom_to_diseases:
                    self._symptom_to_diseases[symptom] = []
                self._symptom_to_diseases[symptom].append(disease_name)
    
    @property
    def disease_count(self) -> int:
        """Кількість діагнозів"""
        return len(self._diseases)
    
    @property
    def symptom_count(self) -> int:
        """Кількість унікальних симптомів"""
        return len(self._all_symptoms)
    
    @property
    def disease_names(self) -> List[str]:
        """Список всіх назв діагнозів"""
        return list(self._diseases.keys())
    
    @property
    def all_symptoms(self) -> List[str]:
        """Відсортований список всіх симптомів"""
        return sorted(self._all_symptoms)
    
    def get_disease(self, name: str) -> Optional[DiseaseRecord]:
        """Отримати запис про діагноз"""
        return self._diseases.get(name)
    
    def get_symptoms(self, disease_name: str) -> List[str]:
        """Отримати симптоми діагнозу"""
        disease = self._diseases.get(disease_name)
        return disease.symptoms if disease else []
    
    def get_diseases_by_symptom(self, symptom: str) -> List[str]:
        """Отримати всі діагнози, що мають цей симптом"""
        symptom = symptom.strip().lower()
        return self._symptom_to_diseases.get(symptom, [])
    
    def has_disease(self, name: str) -> bool:
        """Перевірити чи існує діагноз"""
        return name in self._diseases
    
    def has_symptom(self, symptom: str) -> bool:
        """Перевірити чи існує симптом"""
        return symptom.strip().lower() in self._all_symptoms
    
    def get_statistics(self) -> Dict:
        """Отримати статистику бази даних"""
        symptom_counts = [d.symptom_count for d in self._diseases.values()]
        
        return {
            "disease_count": self.disease_count,
            "symptom_count": self.symptom_count,
            "avg_symptoms_per_disease": sum(symptom_counts) / len(symptom_counts) if symptom_counts else 0,
            "min_symptoms": min(symptom_counts) if symptom_counts else 0,
            "max_symptoms": max(symptom_counts) if symptom_counts else 0,
        }
    
    def __repr__(self) -> str:
        return f"DiseaseDatabaseLoader(diseases={self.disease_count}, symptoms={self.symptom_count})"
