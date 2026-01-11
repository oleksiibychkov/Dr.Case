"""
Dr.Case — Answer Processor

Обробка відповідей пацієнта та оновлення стану діагностики.

Підтримує:
- Crisp відповіді: YES / NO / UNKNOWN
- Заглушка для Fuzzy: "трохи", "сильно" (TODO: нечіткі множини Заде)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .information_gain import (
    InformationGainCalculator,
    AnswerType,
    FuzzyMembershipHandler,
    normalize_probs,
)


@dataclass
class SymptomState:
    """Стан симптому після відповіді"""
    symptom: str
    answer: AnswerType
    membership: Optional[float]  # None для UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_present(self) -> Optional[bool]:
        """Чи симптом присутній (None для UNKNOWN)"""
        if self.answer == AnswerType.YES:
            return True
        elif self.answer == AnswerType.NO:
            return False
        return None
    
    def __repr__(self) -> str:
        return f"SymptomState('{self.symptom}', {self.answer.value})"


@dataclass
class SessionState:
    """
    Повний стан сесії діагностики.
    
    Зберігає:
    - Вектор симптомів
    - Історію питань/відповідей
    - Поточні ймовірності
    """
    # Симптоми
    known_symptoms: Set[str] = field(default_factory=set)      # Підтверджені (YES)
    negated_symptoms: Set[str] = field(default_factory=set)    # Заперечені (NO)
    unknown_symptoms: Set[str] = field(default_factory=set)    # "Не знаю"
    
    # Історія
    symptom_history: List[SymptomState] = field(default_factory=list)
    
    # Ймовірності (оновлюються після кожної відповіді)
    disease_probs: Dict[str, float] = field(default_factory=dict)
    
    # Лічильники
    questions_asked: int = 0
    
    @property
    def all_asked_symptoms(self) -> Set[str]:
        """Всі симптоми про які питали"""
        return self.known_symptoms | self.negated_symptoms | self.unknown_symptoms
    
    def add_symptom_answer(
        self,
        symptom: str,
        answer: AnswerType,
        membership: Optional[float] = None
    ):
        """Додати відповідь на питання про симптом"""
        state = SymptomState(
            symptom=symptom,
            answer=answer,
            membership=membership
        )
        self.symptom_history.append(state)
        self.questions_asked += 1
        
        if answer == AnswerType.YES:
            self.known_symptoms.add(symptom)
        elif answer == AnswerType.NO:
            self.negated_symptoms.add(symptom)
        else:
            self.unknown_symptoms.add(symptom)
    
    def get_symptom_vector(self, symptom_list: List[str]) -> np.ndarray:
        """
        Отримати бінарний вектор симптомів.
        
        Args:
            symptom_list: Повний список симптомів (порядок важливий)
            
        Returns:
            Бінарний вектор [0, 1, 0, 1, ...]
        """
        vector = np.zeros(len(symptom_list), dtype=np.float32)
        
        for i, symptom in enumerate(symptom_list):
            if symptom in self.known_symptoms:
                vector[i] = 1.0
        
        return vector
    
    def to_dict(self) -> dict:
        """Серіалізація стану"""
        return {
            "known_symptoms": list(self.known_symptoms),
            "negated_symptoms": list(self.negated_symptoms),
            "unknown_symptoms": list(self.unknown_symptoms),
            "questions_asked": self.questions_asked,
            "disease_probs": self.disease_probs,
            "history": [
                {
                    "symptom": s.symptom,
                    "answer": s.answer.value,
                    "membership": s.membership,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in self.symptom_history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """Десеріалізація стану"""
        state = cls(
            known_symptoms=set(data.get("known_symptoms", [])),
            negated_symptoms=set(data.get("negated_symptoms", [])),
            unknown_symptoms=set(data.get("unknown_symptoms", [])),
            disease_probs=data.get("disease_probs", {}),
            questions_asked=data.get("questions_asked", 0)
        )
        
        for h in data.get("history", []):
            state.symptom_history.append(SymptomState(
                symptom=h["symptom"],
                answer=AnswerType(h["answer"]),
                membership=h.get("membership"),
                timestamp=datetime.fromisoformat(h["timestamp"])
            ))
        
        return state


class AnswerProcessor:
    """
    Обробник відповідей пацієнта.
    
    Функції:
    1. Парсинг відповідей (текст → AnswerType)
    2. Оновлення ймовірностей (Bayesian update)
    3. Оновлення вектора симптомів
    4. Підтримка fuzzy відповідей (заглушка)
    
    Приклад використання:
        processor = AnswerProcessor(eig_calculator)
        
        # Початковий стан
        state = SessionState()
        state.disease_probs = initial_probs
        
        # Обробити відповідь
        state = processor.process_answer(
            state=state,
            symptom="Muscle Pain",
            answer="yes"
        )
        
        print(state.disease_probs)  # Оновлені ймовірності
    """
    
    def __init__(
        self,
        eig_calculator: InformationGainCalculator,
        fuzzy_enabled: bool = False
    ):
        """
        Args:
            eig_calculator: Калькулятор для Bayesian update
            fuzzy_enabled: Увімкнути нечіткі відповіді (TODO)
        """
        self.eig_calculator = eig_calculator
        self.fuzzy_enabled = fuzzy_enabled
        self.fuzzy_handler = FuzzyMembershipHandler()
        
        # Мапінг текстових відповідей
        self._answer_map = {
            # English
            "yes": AnswerType.YES,
            "no": AnswerType.NO,
            "unknown": AnswerType.UNKNOWN,
            "skip": AnswerType.UNKNOWN,
            "true": AnswerType.YES,
            "false": AnswerType.NO,
            "1": AnswerType.YES,
            "0": AnswerType.NO,
            
            # Ukrainian
            "так": AnswerType.YES,
            "ні": AnswerType.NO,
            "не знаю": AnswerType.UNKNOWN,
            "пропустити": AnswerType.UNKNOWN,
        }
    
    def parse_answer(self, answer: Union[str, bool, int, AnswerType]) -> AnswerType:
        """
        Парсити відповідь у AnswerType.
        
        Args:
            answer: Відповідь у будь-якому форматі
            
        Returns:
            AnswerType
        """
        if isinstance(answer, AnswerType):
            return answer
        
        if isinstance(answer, bool):
            return AnswerType.YES if answer else AnswerType.NO
        
        if isinstance(answer, int):
            if answer == 1:
                return AnswerType.YES
            elif answer == 0:
                return AnswerType.NO
            else:
                return AnswerType.UNKNOWN
        
        if isinstance(answer, str):
            answer_lower = answer.lower().strip()
            return self._answer_map.get(answer_lower, AnswerType.UNKNOWN)
        
        return AnswerType.UNKNOWN
    
    def get_membership(self, answer: AnswerType) -> Optional[float]:
        """
        Отримати membership value для відповіді.
        
        Для crisp логіки:
        - YES → 1.0
        - NO → 0.0
        - UNKNOWN → None (пропускаємо)
        
        TODO: Для fuzzy — значення в діапазоні [0, 1]
        """
        return self.fuzzy_handler.get_membership(answer)
    
    def process_answer(
        self,
        state: SessionState,
        symptom: str,
        answer: Union[str, bool, int, AnswerType]
    ) -> SessionState:
        """
        Обробити відповідь та оновити стан.
        
        Args:
            state: Поточний стан сесії
            symptom: Симптом питання
            answer: Відповідь пацієнта
            
        Returns:
            Оновлений стан
        """
        # Парсимо відповідь
        answer_type = self.parse_answer(answer)
        membership = self.get_membership(answer_type)
        
        # Додаємо в історію
        state.add_symptom_answer(symptom, answer_type, membership)
        
        # Оновлюємо ймовірності (крім UNKNOWN)
        if answer_type != AnswerType.UNKNOWN:
            state.disease_probs = self.eig_calculator.update_probabilities(
                state.disease_probs,
                symptom,
                answer_type
            )
        
        return state
    
    def process_initial_symptoms(
        self,
        symptoms: List[str],
        all_diseases: List[str]
    ) -> SessionState:
        """
        Створити початковий стан з відомими симптомами.
        
        Args:
            symptoms: Список початкових симптомів пацієнта
            all_diseases: Список всіх хвороб
            
        Returns:
            Початковий стан сесії
        """
        state = SessionState()
        
        # Додаємо симптоми як відомі
        for symptom in symptoms:
            state.known_symptoms.add(symptom)
            state.symptom_history.append(SymptomState(
                symptom=symptom,
                answer=AnswerType.YES,
                membership=1.0
            ))
        
        # Початкові рівномірні ймовірності
        # (будуть оновлені після першого прогону NN)
        uniform_prob = 1.0 / len(all_diseases)
        state.disease_probs = {d: uniform_prob for d in all_diseases}
        
        return state
    
    def apply_nn_predictions(
        self,
        state: SessionState,
        nn_probs: Dict[str, float]
    ) -> SessionState:
        """
        Застосувати передбачення NN до стану.
        
        Об'єднує Bayesian прiors з NN predictions.
        
        Args:
            state: Поточний стан
            nn_probs: Ймовірності від NN
            
        Returns:
            Оновлений стан
        """
        # Просто замінюємо (NN вже враховує симптоми)
        # TODO: Можливо, комбінувати Bayesian + NN
        state.disease_probs = nn_probs.copy()
        return state
    
    def get_diagnosis_summary(
        self,
        state: SessionState,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Отримати топ діагнозів з поточного стану.
        
        Args:
            state: Поточний стан
            top_n: Кількість топ діагнозів
            
        Returns:
            [(disease, probability), ...]
        """
        sorted_probs = sorted(
            state.disease_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_probs[:top_n]
    
    def __repr__(self) -> str:
        return (
            f"AnswerProcessor(fuzzy_enabled={self.fuzzy_enabled})"
        )


class SymptomVectorUpdater:
    """
    Оновлення вектора симптомів для NN.
    
    Перетворює SessionState у формат для двогілкової NN:
    - symptom_vector: бінарний вектор [0, 1, ...]
    - Нормалізація: L2
    """
    
    def __init__(self, symptom_names: List[str]):
        """
        Args:
            symptom_names: Список всіх симптомів у правильному порядку
        """
        self.symptom_names = symptom_names
        self.symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}
        self.n_symptoms = len(symptom_names)
    
    def get_vector(
        self,
        state: SessionState,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Отримати вектор симптомів зі стану.
        
        Args:
            state: Стан сесії
            normalize: L2-нормалізація
            
        Returns:
            Вектор симптомів (n_symptoms,)
        """
        vector = np.zeros(self.n_symptoms, dtype=np.float32)
        
        for symptom in state.known_symptoms:
            if symptom in self.symptom_to_idx:
                idx = self.symptom_to_idx[symptom]
                vector[idx] = 1.0
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def add_symptom(
        self,
        vector: np.ndarray,
        symptom: str,
        value: float = 1.0
    ) -> np.ndarray:
        """
        Додати симптом до вектора.
        
        Args:
            vector: Поточний вектор
            symptom: Назва симптому
            value: Значення (1.0 для binary)
            
        Returns:
            Оновлений вектор
        """
        if symptom in self.symptom_to_idx:
            idx = self.symptom_to_idx[symptom]
            vector = vector.copy()
            vector[idx] = value
        return vector
    
    def remove_symptom(
        self,
        vector: np.ndarray,
        symptom: str
    ) -> np.ndarray:
        """Видалити симптом з вектора"""
        return self.add_symptom(vector, symptom, value=0.0)
    
    def __repr__(self) -> str:
        return f"SymptomVectorUpdater(symptoms={self.n_symptoms})"
