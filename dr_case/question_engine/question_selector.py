"""
Dr.Case — Question Selector

Вибір найкращого уточнюючого питання на основі EIG.

Питання = запит про наявність симптому.
Мета = максимально зменшити невизначеність між топ-кандидатами.
"""

import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

from .information_gain import (
    InformationGainCalculator,
    EIGResult,
    AnswerType,
)


@dataclass
class Question:
    """Уточнююче питання для пацієнта"""
    symptom: str
    text: str  # Текст питання для відображення
    eig: float
    p_yes: float
    p_no: float
    
    # Додаткова інформація
    discriminates: List[Tuple[str, str]] = field(default_factory=list)  # Які хвороби розрізняє
    
    def __repr__(self) -> str:
        return f"Question('{self.symptom}', eig={self.eig:.4f})"


@dataclass 
class QuestionResult:
    """Результат відповіді на питання"""
    question: Question
    answer: AnswerType
    membership: Optional[float]  # Для fuzzy (None для UNKNOWN)
    
    # Зміни після відповіді
    probs_before: Dict[str, float] = field(default_factory=dict)
    probs_after: Dict[str, float] = field(default_factory=dict)
    entropy_reduction: float = 0.0


class QuestionGenerator:
    """
    Генератор текстів питань.
    
    Перетворює назву симптому на зрозуміле питання.
    Підтримує українську та англійську мови.
    """
    
    def __init__(self, language: str = "uk"):
        """
        Args:
            language: Мова питань ("uk" або "en")
        """
        self.language = language
        
        # Шаблони питань
        self._templates = {
            "uk": {
                "default": "Чи є у вас {symptom}?",
                "pain": "Чи відчуваєте ви біль: {symptom}?",
                "feeling": "Чи відчуваєте ви {symptom}?",
            },
            "en": {
                "default": "Do you have {symptom}?",
                "pain": "Do you experience {symptom}?",
                "feeling": "Are you feeling {symptom}?",
            }
        }
        
        # Переклади симптомів (базові)
        self._symptom_translations = {
            # Pain
            "Headache": {"uk": "головний біль", "en": "headache"},
            "Chest Pain": {"uk": "біль у грудях", "en": "chest pain"},
            "Abdominal Pain": {"uk": "біль у животі", "en": "abdominal pain"},
            "Muscle Pain": {"uk": "біль у м'язах", "en": "muscle pain"},
            "Joint Pain": {"uk": "біль у суглобах", "en": "joint pain"},
            "Back Pain": {"uk": "біль у спині", "en": "back pain"},
            "Sore Throat": {"uk": "біль у горлі", "en": "sore throat"},
            
            # General
            "Fever": {"uk": "підвищена температура", "en": "fever"},
            "Fatigue": {"uk": "втома", "en": "fatigue"},
            "Weakness": {"uk": "слабкість", "en": "weakness"},
            "Chills": {"uk": "озноб", "en": "chills"},
            "Sweating": {"uk": "пітливість", "en": "sweating"},
            "Weight Loss": {"uk": "втрата ваги", "en": "weight loss"},
            
            # Respiratory
            "Cough": {"uk": "кашель", "en": "cough"},
            "Shortness of Breath": {"uk": "задишка", "en": "shortness of breath"},
            "Runny Nose": {"uk": "нежить", "en": "runny nose"},
            "Nasal Congestion": {"uk": "закладеність носа", "en": "nasal congestion"},
            "Sneezing": {"uk": "чхання", "en": "sneezing"},
            
            # GI
            "Nausea": {"uk": "нудота", "en": "nausea"},
            "Vomiting": {"uk": "блювання", "en": "vomiting"},
            "Diarrhea": {"uk": "діарея", "en": "diarrhea"},
            "Constipation": {"uk": "закреп", "en": "constipation"},
            "Loss of Appetite": {"uk": "втрата апетиту", "en": "loss of appetite"},
            
            # Neuro
            "Dizziness": {"uk": "запаморочення", "en": "dizziness"},
            "Confusion": {"uk": "сплутаність свідомості", "en": "confusion"},
            "Memory Loss": {"uk": "втрата пам'яті", "en": "memory loss"},
            
            # Skin
            "Rash": {"uk": "висип", "en": "rash"},
            "Itching": {"uk": "свербіж", "en": "itching"},
            
            # COVID-specific
            "Loss of Smell": {"uk": "втрата нюху", "en": "loss of smell"},
            "Loss of Taste": {"uk": "втрата смаку", "en": "loss of taste"},
        }
    
    def translate_symptom(self, symptom: str) -> str:
        """Перекласти назву симптому"""
        if symptom in self._symptom_translations:
            return self._symptom_translations[symptom].get(
                self.language, symptom.lower()
            )
        # Fallback: lowercase
        return symptom.lower().replace("_", " ")
    
    def generate_text(self, symptom: str) -> str:
        """
        Згенерувати текст питання для симптому.
        
        Args:
            symptom: Назва симптому (англійською)
            
        Returns:
            Текст питання на обраній мові
        """
        translated = self.translate_symptom(symptom)
        template = self._templates[self.language]["default"]
        return template.format(symptom=translated)
    
    def add_translation(self, symptom: str, translations: Dict[str, str]):
        """Додати переклад для симптому"""
        self._symptom_translations[symptom] = translations


class QuestionSelector:
    """
    Вибір найкращого уточнюючого питання.
    
    Інтегрує:
    - InformationGainCalculator для обчислення EIG
    - QuestionGenerator для генерації тексту питань
    - Логіку фільтрації та пріоритизації
    
    Приклад використання:
        selector = QuestionSelector.from_database("data/database.json")
        
        # Поточні ймовірності від NN
        probs = {"Influenza": 0.45, "Cold": 0.32, "COVID": 0.15}
        
        # Відомі симптоми
        known = {"Fever", "Cough"}
        
        # Отримати питання
        question = selector.select_question(probs, known_symptoms=known)
        
        if question:
            print(f"Питання: {question.text}")
            print(f"EIG: {question.eig:.4f}")
    """
    
    def __init__(
        self,
        eig_calculator: InformationGainCalculator,
        question_generator: Optional[QuestionGenerator] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            eig_calculator: Калькулятор Information Gain
            question_generator: Генератор тексту питань
            config: Конфігурація
        """
        self.eig_calculator = eig_calculator
        self.question_generator = question_generator or QuestionGenerator()
        
        # Конфігурація за замовчуванням
        self.config = {
            "min_disease_prob": 0.05,      # Мін. ймовірність хвороби
            "min_eig_threshold": 0.001,    # Мін. EIG для питання
            "max_questions": 20,            # Макс. питань за сесію
            "prefer_easy_questions": True,  # Пріоритет простим питанням
            "prefer_discriminative": True,  # Пріоритет питанням що розрізняють топ
        }
        
        if config:
            self.config.update(config)
    
    @classmethod
    def from_database(
        cls,
        database_path: str,
        language: str = "uk",
        **config
    ) -> "QuestionSelector":
        """
        Створити з бази даних.
        
        Args:
            database_path: Шлях до JSON бази
            language: Мова питань
            **config: Додаткові параметри конфігурації
        """
        eig_calc = InformationGainCalculator.from_json_file(database_path)
        generator = QuestionGenerator(language=language)
        return cls(eig_calc, generator, config)
    
    @classmethod
    def from_loaded_database(
        cls,
        database: Dict[str, dict],
        language: str = "uk",
        **config
    ) -> "QuestionSelector":
        """Створити з вже завантаженої бази даних"""
        eig_calc = InformationGainCalculator.from_database(database)
        generator = QuestionGenerator(language=language)
        return cls(eig_calc, generator, config)
    
    def select_question(
        self,
        disease_probs: Dict[str, float],
        known_symptoms: Optional[Set[str]] = None,
        asked_symptoms: Optional[Set[str]] = None,
        negated_symptoms: Optional[Set[str]] = None,
    ) -> Optional[Question]:
        """
        Вибрати найкраще питання.
        
        Args:
            disease_probs: Поточні ймовірності {disease: prob} від NN
            known_symptoms: Симптоми, які пацієнт вже підтвердив
            asked_symptoms: Симптоми, про які вже питали (включаючи "ні")
            negated_symptoms: Симптоми, які пацієнт заперечив
            
        Returns:
            Question або None якщо немає хороших питань
        """
        # Об'єднуємо всі виключення
        exclude = set()
        if known_symptoms:
            exclude.update(known_symptoms)
        if asked_symptoms:
            exclude.update(asked_symptoms)
        if negated_symptoms:
            exclude.update(negated_symptoms)
        
        # Обчислюємо EIG для всіх кандидатів
        eig_results = self.eig_calculator.compute_all_eig(
            disease_probs=disease_probs,
            exclude_symptoms=exclude,
            min_prob_threshold=self.config["min_disease_prob"],
            top_k=10  # Беремо топ-10 для додаткової фільтрації
        )
        
        if not eig_results:
            return None
        
        # Фільтруємо за мінімальним EIG
        min_eig = self.config["min_eig_threshold"]
        eig_results = [r for r in eig_results if r.eig >= min_eig]
        
        if not eig_results:
            return None
        
        # Вибираємо найкращий
        best = eig_results[0]
        
        # Генеруємо текст питання
        question_text = self.question_generator.generate_text(best.symptom)
        
        # Знаходимо які хвороби розрізняє це питання
        discriminates = self._find_discriminated_diseases(
            best.symptom, disease_probs
        )
        
        return Question(
            symptom=best.symptom,
            text=question_text,
            eig=best.eig,
            p_yes=best.p_yes,
            p_no=best.p_no,
            discriminates=discriminates
        )
    
    def select_top_questions(
        self,
        disease_probs: Dict[str, float],
        known_symptoms: Optional[Set[str]] = None,
        asked_symptoms: Optional[Set[str]] = None,
        top_k: int = 5
    ) -> List[Question]:
        """
        Вибрати топ-K питань.
        
        Корисно для відображення альтернатив або batch-запитів.
        """
        exclude = set()
        if known_symptoms:
            exclude.update(known_symptoms)
        if asked_symptoms:
            exclude.update(asked_symptoms)
        
        eig_results = self.eig_calculator.compute_all_eig(
            disease_probs=disease_probs,
            exclude_symptoms=exclude,
            min_prob_threshold=self.config["min_disease_prob"],
            top_k=top_k
        )
        
        questions = []
        for result in eig_results:
            if result.eig >= self.config["min_eig_threshold"]:
                question_text = self.question_generator.generate_text(result.symptom)
                discriminates = self._find_discriminated_diseases(
                    result.symptom, disease_probs
                )
                
                questions.append(Question(
                    symptom=result.symptom,
                    text=question_text,
                    eig=result.eig,
                    p_yes=result.p_yes,
                    p_no=result.p_no,
                    discriminates=discriminates
                ))
        
        return questions
    
    def _find_discriminated_diseases(
        self,
        symptom: str,
        disease_probs: Dict[str, float],
        top_n: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Знайти пари хвороб, які розрізняє симптом.
        
        Returns:
            [(disease1, disease2), ...] де symptom має найбільшу різницю частот
        """
        # Беремо топ хвороби
        sorted_diseases = sorted(
            disease_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        pairs = []
        for i, (d1, _) in enumerate(sorted_diseases):
            for d2, _ in sorted_diseases[i+1:]:
                f1 = self.eig_calculator.get_symptom_probability(d1, symptom)
                f2 = self.eig_calculator.get_symptom_probability(d2, symptom)
                diff = abs(f1 - f2)
                if diff > 0.3:  # Значна різниця
                    pairs.append((d1, d2, diff))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return [(d1, d2) for d1, d2, _ in pairs[:top_n]]
    
    def process_answer(
        self,
        disease_probs: Dict[str, float],
        symptom: str,
        answer: AnswerType
    ) -> Dict[str, float]:
        """
        Обробити відповідь та оновити ймовірності.
        
        Args:
            disease_probs: Поточні ймовірності
            symptom: Симптом питання
            answer: Відповідь пацієнта
            
        Returns:
            Оновлені ймовірності
        """
        return self.eig_calculator.update_probabilities(
            disease_probs, symptom, answer
        )
    
    def explain_question(
        self,
        question: Question,
        disease_probs: Dict[str, float],
        top_n: int = 3
    ) -> str:
        """
        Пояснити чому обрано це питання.
        
        Args:
            question: Питання
            disease_probs: Поточні ймовірності
            top_n: Кількість хвороб у поясненні
            
        Returns:
            Текстове пояснення
        """
        # Топ хвороби
        sorted_diseases = sorted(
            disease_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        lines = []
        lines.append(f"Питання про '{question.symptom}' допоможе розрізнити:")
        
        for disease, prob in sorted_diseases:
            freq = self.eig_calculator.get_symptom_probability(disease, question.symptom)
            lines.append(f"  • {disease} ({prob:.0%}): симптом у {freq:.0%} випадків")
        
        lines.append(f"\nОчікуване зменшення невизначеності: {question.eig:.3f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"QuestionSelector(\n"
            f"  {self.eig_calculator},\n"
            f"  language='{self.question_generator.language}',\n"
            f"  min_disease_prob={self.config['min_disease_prob']}\n"
            f")"
        )
