"""
Dr.Case — Information Gain для вибору питань

Обчислення Expected Information Gain (EIG) для симптомів.

EIG(q) = H(ŷ) - E[H(ŷ|answer)]

де:
- H(ŷ) = ентропія поточного розподілу ймовірностей діагнозів
- E[H(ŷ|answer)] = очікувана ентропія після отримання відповіді

Використовуємо реальні частоти P(symptom|disease) з symptom_frequency бази даних.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class AnswerType(Enum):
    """Типи відповідей на питання"""
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"  # "Не знаю"


@dataclass
class EIGResult:
    """Результат обчислення EIG для симптому"""
    symptom: str
    eig: float
    p_yes: float  # P(symptom = yes)
    p_no: float   # P(symptom = no)
    h_current: float  # поточна ентропія
    h_after_yes: float  # ентропія якщо відповідь "так"
    h_after_no: float   # ентропія якщо відповідь "ні"
    
    def __repr__(self) -> str:
        return (
            f"EIGResult(symptom='{self.symptom}', eig={self.eig:.4f}, "
            f"p_yes={self.p_yes:.2%}, p_no={self.p_no:.2%})"
        )


def entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """
    Обчислити ентропію Шеннона.
    
    H = -Σ p_i * log(p_i)
    
    Args:
        probs: Масив ймовірностей (має сумуватись до 1)
        eps: Мала константа для уникнення log(0)
        
    Returns:
        Ентропія в натуральних одиницях (nats)
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


def normalize_probs(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Нормалізувати ймовірності щоб сума = 1"""
    probs = np.asarray(probs, dtype=np.float64)
    total = np.sum(probs)
    if total < eps:
        return np.ones_like(probs) / len(probs)
    return probs / total


class FuzzyMembershipHandler:
    """
    Заглушка для обробки нечітких відповідей (Fuzzy Sets Zadeh).
    
    TODO: Реалізувати нечіткі множини для обробки відповідей типу:
    - "Трохи болить" → membership = 0.3
    - "Сильно болить" → membership = 0.9
    - "Не знаю" → membership = 0.5 або спеціальна обробка
    
    Поки що використовується crisp (чітка) логіка:
    - YES = 1.0
    - NO = 0.0
    - UNKNOWN = skip (не оновлюємо)
    """
    
    def __init__(self):
        """Ініціалізація обробника нечітких відповідей"""
        # Параметри для майбутньої реалізації
        self.fuzzy_enabled = False
        
        # Membership functions (заглушки)
        self._membership_functions = {
            "none": lambda x: 0.0,
            "mild": lambda x: 0.3,
            "moderate": lambda x: 0.6,
            "severe": lambda x: 0.9,
            "present": lambda x: 1.0,
        }
    
    def get_membership(self, answer: Union[AnswerType, str, float]) -> Optional[float]:
        """
        Отримати значення membership для відповіді.
        
        Args:
            answer: Відповідь (AnswerType, строка або число 0-1)
            
        Returns:
            Membership value [0, 1] або None для UNKNOWN
        """
        if isinstance(answer, (int, float)):
            return float(np.clip(answer, 0.0, 1.0))
        
        if isinstance(answer, AnswerType):
            if answer == AnswerType.YES:
                return 1.0
            elif answer == AnswerType.NO:
                return 0.0
            elif answer == AnswerType.UNKNOWN:
                return None  # Пропускаємо
        
        if isinstance(answer, str):
            answer_lower = answer.lower()
            if answer_lower in ("yes", "так", "true", "1"):
                return 1.0
            elif answer_lower in ("no", "ні", "false", "0"):
                return 0.0
            elif answer_lower in ("unknown", "не знаю", "?", "skip"):
                return None
            # Fuzzy strings (для майбутньої реалізації)
            elif answer_lower in self._membership_functions:
                return self._membership_functions[answer_lower](None)
        
        return None
    
    def update_probabilities_fuzzy(
        self,
        disease_probs: Dict[str, float],
        symptom: str,
        membership: float,
        symptom_frequencies: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Оновити ймовірності з урахуванням нечіткого membership.
        
        TODO: Реалізувати повноцінне Bayesian оновлення з fuzzy evidence.
        
        Поки що: лінійна інтерполяція між P(D|S=yes) та P(D|S=no)
        
        Args:
            disease_probs: Поточні ймовірності {disease: prob}
            symptom: Назва симптому
            membership: Значення membership [0, 1]
            symptom_frequencies: Частоти симптомів {disease: {symptom: freq}}
            
        Returns:
            Оновлені ймовірності
        """
        # Заглушка: crisp оновлення
        # TODO: Замінити на fuzzy Bayesian inference
        
        if membership is None:
            return disease_probs.copy()
        
        if membership > 0.5:
            # Трактуємо як YES
            return self._update_crisp(disease_probs, symptom, True, symptom_frequencies)
        else:
            # Трактуємо як NO
            return self._update_crisp(disease_probs, symptom, False, symptom_frequencies)
    
    def _update_crisp(
        self,
        disease_probs: Dict[str, float],
        symptom: str,
        is_present: bool,
        symptom_frequencies: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Crisp (чітке) Bayesian оновлення"""
        updated = {}
        
        for disease, prior in disease_probs.items():
            freq = symptom_frequencies.get(disease, {}).get(symptom, 0.1)
            
            if is_present:
                likelihood = freq
            else:
                likelihood = 1.0 - freq
            
            updated[disease] = prior * likelihood
        
        # Нормалізуємо
        total = sum(updated.values())
        if total > 0:
            updated = {d: p / total for d, p in updated.items()}
        
        return updated


class InformationGainCalculator:
    """
    Калькулятор Expected Information Gain.
    
    Використовує частоти P(symptom|disease) з symptom_frequency бази даних
    для обчислення EIG кожного потенційного питання.
    
    Приклад використання:
        calculator = InformationGainCalculator.from_database(database)
        
        # Поточні ймовірності від NN
        current_probs = {"Influenza": 0.45, "Cold": 0.32, "COVID": 0.15}
        
        # Знайти найкраще питання
        best = calculator.select_best_question(
            disease_probs=current_probs,
            asked_symptoms={"Fever", "Cough"},
            min_disease_prob=0.05
        )
        
        print(f"Питання: {best.symptom}, EIG = {best.eig:.4f}")
    """
    
    def __init__(
        self,
        symptom_frequencies: Dict[str, Dict[str, float]],
        default_frequency: float = 0.1,
        min_disease_prob: float = 0.05,
    ):
        """
        Args:
            symptom_frequencies: {disease: {symptom: frequency}}
                frequency = P(symptom|disease) в діапазоні [0, 1]
            default_frequency: Частота за замовчуванням якщо симптом невідомий
            min_disease_prob: Мінімальна ймовірність хвороби для врахування (параметр)
        """
        self.symptom_frequencies = symptom_frequencies
        self.default_frequency = default_frequency
        self.min_disease_prob = min_disease_prob
        
        # Обробник нечітких відповідей
        self.fuzzy_handler = FuzzyMembershipHandler()
        
        # Кешуємо всі унікальні симптоми
        self._all_symptoms = set()
        for disease_symptoms in symptom_frequencies.values():
            self._all_symptoms.update(disease_symptoms.keys())
        
        # Кешуємо всі хвороби
        self._all_diseases = set(symptom_frequencies.keys())
    
    @classmethod
    def from_database(cls, database: Dict[str, dict], **kwargs) -> "InformationGainCalculator":
        """
        Створити з бази даних.
        
        Args:
            database: Завантажена JSON база {disease: {symptoms, symptom_frequency, ...}}
            **kwargs: Додаткові параметри (default_frequency, min_disease_prob)
            
        Returns:
            InformationGainCalculator
        """
        symptom_frequencies = {}
        
        for disease, data in database.items():
            freq_dict = data.get("symptom_frequency", {})
            
            if freq_dict:
                # Нормалізуємо до [0, 1]
                # Частоти в базі можуть бути в % (0-100) або вже нормалізовані
                values = list(freq_dict.values())
                max_freq = max(values) if values else 1.0
                
                # Якщо max > 1, значить частоти в % — нормалізуємо
                if max_freq > 1.0:
                    symptom_frequencies[disease] = {
                        symptom: freq / 100.0 
                        for symptom, freq in freq_dict.items()
                    }
                else:
                    symptom_frequencies[disease] = freq_dict.copy()
            else:
                # Якщо немає частот - використовуємо бінарні (є/немає = 1.0)
                symptoms = data.get("symptoms", [])
                symptom_frequencies[disease] = {s: 1.0 for s in symptoms}
        
        return cls(symptom_frequencies, **kwargs)
    
    @classmethod
    def from_json_file(cls, path: str, **kwargs) -> "InformationGainCalculator":
        """Створити з JSON файлу"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        return cls.from_database(database, **kwargs)
    
    def get_symptom_probability(self, disease: str, symptom: str) -> float:
        """
        Отримати P(symptom|disease).
        
        Args:
            disease: Назва хвороби
            symptom: Назва симптому
            
        Returns:
            Ймовірність симптому при даній хворобі [0, 1]
        """
        disease_symptoms = self.symptom_frequencies.get(disease, {})
        return disease_symptoms.get(symptom, self.default_frequency)
    
    def compute_eig(
        self,
        symptom: str,
        disease_probs: Dict[str, float],
        min_prob_threshold: Optional[float] = None
    ) -> EIGResult:
        """
        Обчислити Expected Information Gain для симптому.
        
        EIG = H(current) - [P(yes) * H(after_yes) + P(no) * H(after_no)]
        
        Args:
            symptom: Симптом для оцінки
            disease_probs: Поточні ймовірності {disease: prob}
            min_prob_threshold: Мінімальний поріг (None = використати self.min_disease_prob)
            
        Returns:
            EIGResult з деталями обчислення
        """
        threshold = min_prob_threshold if min_prob_threshold is not None else self.min_disease_prob
        
        # Фільтруємо хвороби за порогом
        filtered_diseases = {
            d: p for d, p in disease_probs.items() 
            if p >= threshold
        }
        
        # Якщо нічого не залишилось — беремо всі
        if not filtered_diseases:
            filtered_diseases = disease_probs.copy()
        
        diseases = list(filtered_diseases.keys())
        probs = np.array([filtered_diseases[d] for d in diseases])
        probs = normalize_probs(probs)
        
        # Поточна ентропія
        h_current = entropy(probs)
        
        # P(symptom|disease) для кожної хвороби
        symptom_given_disease = np.array([
            self.get_symptom_probability(d, symptom) for d in diseases
        ])
        
        # P(symptom = yes) = Σ P(disease) * P(symptom|disease)
        p_yes = float(np.sum(probs * symptom_given_disease))
        p_no = 1.0 - p_yes
        
        eps = 1e-10
        
        # Байєсівське оновлення для відповіді "так"
        # P(disease|symptom=yes) ∝ P(symptom|disease) * P(disease)
        if p_yes > eps:
            probs_after_yes = probs * symptom_given_disease
            probs_after_yes = normalize_probs(probs_after_yes)
            h_after_yes = entropy(probs_after_yes)
        else:
            h_after_yes = h_current
        
        # Байєсівське оновлення для відповіді "ні"
        # P(disease|symptom=no) ∝ P(no_symptom|disease) * P(disease)
        if p_no > eps:
            probs_after_no = probs * (1.0 - symptom_given_disease)
            probs_after_no = normalize_probs(probs_after_no)
            h_after_no = entropy(probs_after_no)
        else:
            h_after_no = h_current
        
        # Expected Information Gain
        expected_h_after = p_yes * h_after_yes + p_no * h_after_no
        eig = h_current - expected_h_after
        
        return EIGResult(
            symptom=symptom,
            eig=max(0.0, eig),  # EIG не може бути від'ємним
            p_yes=p_yes,
            p_no=p_no,
            h_current=h_current,
            h_after_yes=h_after_yes,
            h_after_no=h_after_no
        )
    
    def compute_all_eig(
        self,
        disease_probs: Dict[str, float],
        exclude_symptoms: Optional[set] = None,
        min_prob_threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[EIGResult]:
        """
        Обчислити EIG для всіх доступних симптомів.
        
        Args:
            disease_probs: Поточні ймовірності діагнозів
            exclude_symptoms: Симптоми для виключення (вже відомі/запитані)
            min_prob_threshold: Мінімальний поріг ймовірності хвороби
            top_k: Повернути тільки топ-k за EIG
            
        Returns:
            Список EIGResult, відсортований за EIG (спадно)
        """
        threshold = min_prob_threshold if min_prob_threshold is not None else self.min_disease_prob
        exclude = exclude_symptoms or set()
        
        # Збираємо симптоми з релевантних хвороб
        relevant_symptoms = set()
        for disease, prob in disease_probs.items():
            if prob >= threshold:
                disease_symptoms = self.symptom_frequencies.get(disease, {})
                relevant_symptoms.update(disease_symptoms.keys())
        
        # Фільтруємо виключені
        candidate_symptoms = relevant_symptoms - exclude
        
        # Обчислюємо EIG
        results = []
        for symptom in candidate_symptoms:
            result = self.compute_eig(symptom, disease_probs, threshold)
            results.append(result)
        
        # Сортуємо за EIG (спадно)
        results.sort(key=lambda r: r.eig, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def select_best_question(
        self,
        disease_probs: Dict[str, float],
        asked_symptoms: Optional[set] = None,
        known_symptoms: Optional[set] = None,
        min_prob_threshold: Optional[float] = None,
        min_eig_threshold: float = 0.001
    ) -> Optional[EIGResult]:
        """
        Вибрати найкраще питання (симптом з максимальним EIG).
        
        Args:
            disease_probs: Поточні ймовірності діагнозів від NN
            asked_symptoms: Симптоми, які вже запитували
            known_symptoms: Симптоми, які вже відомі (позитивні)
            min_prob_threshold: Мінімальна ймовірність хвороби для врахування
            min_eig_threshold: Мінімальний EIG для повернення питання
            
        Returns:
            EIGResult для найкращого питання або None
        """
        # Об'єднуємо виключення
        exclude = set()
        if asked_symptoms:
            exclude.update(asked_symptoms)
        if known_symptoms:
            exclude.update(known_symptoms)
        
        # Отримуємо топ-1
        results = self.compute_all_eig(
            disease_probs=disease_probs,
            exclude_symptoms=exclude,
            min_prob_threshold=min_prob_threshold,
            top_k=1
        )
        
        if not results:
            return None
        
        best = results[0]
        
        # Перевіряємо мінімальний поріг EIG
        if best.eig < min_eig_threshold:
            return None
        
        return best
    
    def update_probabilities(
        self,
        disease_probs: Dict[str, float],
        symptom: str,
        answer: Union[AnswerType, str, float, bool]
    ) -> Dict[str, float]:
        """
        Оновити ймовірності після відповіді на питання.
        
        Args:
            disease_probs: Поточні ймовірності {disease: prob}
            symptom: Симптом, про який запитували
            answer: Відповідь (YES/NO/UNKNOWN, bool, або float для fuzzy)
            
        Returns:
            Оновлені ймовірності {disease: prob}
        """
        # Конвертуємо bool
        if isinstance(answer, bool):
            answer = AnswerType.YES if answer else AnswerType.NO
        
        # Отримуємо membership через fuzzy handler
        membership = self.fuzzy_handler.get_membership(answer)
        
        # Якщо UNKNOWN — пропускаємо (не оновлюємо)
        if membership is None:
            return disease_probs.copy()
        
        # Bayesian оновлення
        updated = {}
        
        for disease, prior in disease_probs.items():
            freq = self.get_symptom_probability(disease, symptom)
            
            # Для fuzzy: інтерполяція між YES та NO likelihood
            # likelihood = membership * freq + (1 - membership) * (1 - freq)
            if membership >= 0.5:
                # Більше схоже на YES
                likelihood = freq
            else:
                # Більше схоже на NO
                likelihood = 1.0 - freq
            
            # Повна fuzzy версія (TODO: увімкнути коли fuzzy_enabled)
            # likelihood = membership * freq + (1 - membership) * (1 - freq)
            
            updated[disease] = prior * max(likelihood, 1e-10)
        
        # Нормалізуємо
        total = sum(updated.values())
        if total > 0:
            updated = {d: p / total for d, p in updated.items()}
        else:
            updated = disease_probs.copy()
        
        return updated
    
    @property
    def all_symptoms(self) -> set:
        """Всі відомі симптоми"""
        return self._all_symptoms.copy()
    
    @property
    def all_diseases(self) -> set:
        """Всі відомі хвороби"""
        return self._all_diseases.copy()
    
    def get_discriminative_symptoms(
        self,
        disease1: str,
        disease2: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        Знайти симптоми, що найкраще розрізняють дві хвороби.
        
        Args:
            disease1, disease2: Хвороби для порівняння
            top_k: Кількість симптомів
            
        Returns:
            [(symptom, freq1, freq2), ...] відсортовано за |freq1 - freq2|
        """
        freq1 = self.symptom_frequencies.get(disease1, {})
        freq2 = self.symptom_frequencies.get(disease2, {})
        
        all_symptoms = set(freq1.keys()) | set(freq2.keys())
        
        diffs = []
        for symptom in all_symptoms:
            f1 = freq1.get(symptom, self.default_frequency)
            f2 = freq2.get(symptom, self.default_frequency)
            diff = abs(f1 - f2)
            diffs.append((symptom, f1, f2, diff))
        
        diffs.sort(key=lambda x: x[3], reverse=True)
        
        return [(s, f1, f2) for s, f1, f2, _ in diffs[:top_k]]
    
    def __repr__(self) -> str:
        return (
            f"InformationGainCalculator(\n"
            f"  diseases={len(self._all_diseases)},\n"
            f"  symptoms={len(self._all_symptoms)},\n"
            f"  min_disease_prob={self.min_disease_prob},\n"
            f"  default_frequency={self.default_frequency}\n"
            f")"
        )
