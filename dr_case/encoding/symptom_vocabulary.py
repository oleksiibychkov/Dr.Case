"""
Dr.Case — Словник симптомів

Відображення симптомів у числові індекси та навпаки.
Це основа для векторизації.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from .data_loader import DiseaseDatabaseLoader


class SymptomVocabulary:
    """
    Словник симптомів: symptom ↔ index
    
    Приклад використання:
        vocab = SymptomVocabulary.from_database("data/unified_disease_symptom_data_full.json")
        
        # Симптом → індекс
        idx = vocab.symptom_to_index("fever")
        
        # Індекс → симптом
        symptom = vocab.index_to_symptom(42)
        
        # Розмір словника
        print(vocab.size)  # 461
    """
    
    def __init__(self):
        self._symptom_to_idx: Dict[str, int] = {}
        self._idx_to_symptom: Dict[int, str] = {}
        self._frozen: bool = False
    
    @classmethod
    def from_database(cls, path: Optional[str] = None) -> "SymptomVocabulary":
        """
        Створити словник з бази даних.
        
        Args:
            path: Шлях до JSON файлу бази даних
            
        Returns:
            Заповнений SymptomVocabulary
        """
        loader = DiseaseDatabaseLoader(path)
        vocab = cls()
        
        # Додаємо всі симптоми в алфавітному порядку
        for symptom in loader.all_symptoms:
            vocab.add_symptom(symptom)
        
        vocab.freeze()
        return vocab
    
    @classmethod
    def from_symptoms(cls, symptoms: List[str]) -> "SymptomVocabulary":
        """
        Створити словник зі списку симптомів.
        
        Args:
            symptoms: Список симптомів
            
        Returns:
            Заповнений SymptomVocabulary
        """
        vocab = cls()
        
        # Сортуємо для детермінованості
        for symptom in sorted(set(symptoms)):
            vocab.add_symptom(symptom)
        
        vocab.freeze()
        return vocab
    
    def add_symptom(self, symptom: str) -> int:
        """
        Додати симптом до словника.
        
        Args:
            symptom: Назва симптому
            
        Returns:
            Індекс симптому
        """
        if self._frozen:
            raise RuntimeError("Vocabulary is frozen. Cannot add new symptoms.")
        
        symptom = symptom.strip().lower()
        
        if symptom in self._symptom_to_idx:
            return self._symptom_to_idx[symptom]
        
        idx = len(self._symptom_to_idx)
        self._symptom_to_idx[symptom] = idx
        self._idx_to_symptom[idx] = symptom
        
        return idx
    
    def freeze(self) -> None:
        """Заморозити словник (заборонити додавання нових симптомів)"""
        self._frozen = True
    
    def symptom_to_index(self, symptom: str) -> Optional[int]:
        """
        Отримати індекс симптому.
        
        Args:
            symptom: Назва симптому
            
        Returns:
            Індекс або None якщо симптом не знайдено
        """
        symptom = symptom.strip().lower()
        return self._symptom_to_idx.get(symptom)
    
    def index_to_symptom(self, index: int) -> Optional[str]:
        """
        Отримати симптом за індексом.
        
        Args:
            index: Індекс симптому
            
        Returns:
            Назва симптому або None
        """
        return self._idx_to_symptom.get(index)
    
    def has_symptom(self, symptom: str) -> bool:
        """Перевірити чи є симптом у словнику"""
        return symptom.strip().lower() in self._symptom_to_idx
    
    @property
    def size(self) -> int:
        """Кількість симптомів у словнику"""
        return len(self._symptom_to_idx)
    
    @property
    def symptoms(self) -> List[str]:
        """Список всіх симптомів (в порядку індексів)"""
        return [self._idx_to_symptom[i] for i in range(self.size)]
    
    def get_indices(self, symptoms: List[str]) -> List[int]:
        """
        Отримати індекси для списку симптомів.
        
        Args:
            symptoms: Список симптомів
            
        Returns:
            Список індексів (тільки для знайдених симптомів)
        """
        indices = []
        for symptom in symptoms:
            idx = self.symptom_to_index(symptom)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def save(self, path: str) -> None:
        """
        Зберегти словник у JSON файл.
        
        Args:
            path: Шлях до файлу
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "symptom_to_index": self._symptom_to_idx,
            "size": self.size,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "SymptomVocabulary":
        """
        Завантажити словник з JSON файлу.
        
        Args:
            path: Шлях до файлу
            
        Returns:
            SymptomVocabulary
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = cls()
        vocab._symptom_to_idx = data["symptom_to_index"]
        vocab._idx_to_symptom = {int(v): k for k, v in vocab._symptom_to_idx.items()}
        vocab.freeze()
        
        return vocab
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, symptom: str) -> bool:
        return self.has_symptom(symptom)
    
    def __repr__(self) -> str:
        return f"SymptomVocabulary(size={self.size}, frozen={self._frozen})"
