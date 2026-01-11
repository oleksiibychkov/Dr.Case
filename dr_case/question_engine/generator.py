"""
Dr.Case — Генератор уточнюючих питань

QuestionGenerator вибирає найкращі питання для уточнення діагнозу
на основі Information Gain та інших критеріїв.

Стратегії вибору питань:
1. Max Information Gain — максимально зменшує невизначеність
2. Balanced Split — рівномірно розділяє кандидатів
3. Safety First — спочатку питання про небезпечні симптоми
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from dr_case.schemas import Question
from dr_case.encoding import DiseaseEncoder, SymptomVocabulary
from .information_gain import InformationGainCalculator, SymptomInformationGain


class QuestionStrategy(Enum):
    """Стратегія вибору питань"""
    MAX_INFO_GAIN = "max_info_gain"
    BALANCED_SPLIT = "balanced_split"
    SAFETY_FIRST = "safety_first"
    HYBRID = "hybrid"


@dataclass
class QuestionCandidate:
    """Кандидат на питання"""
    symptom: str
    information_gain: float
    split_ratio: float
    
    # Додаткові метрики
    is_safety_critical: bool = False
    priority_score: float = 0.0
    
    # Пояснення
    diseases_if_yes: List[str] = field(default_factory=list)
    diseases_if_no: List[str] = field(default_factory=list)
    
    def to_question(self) -> Question:
        """Конвертувати в Question schema"""
        return Question(
            symptom=self.symptom,
            text=self._generate_text(),
            information_gain=self.information_gain,
            explanation=self._generate_explanation()
        )
    
    def _generate_text(self) -> str:
        """Згенерувати текст питання"""
        # Базовий шаблон
        symptom_text = self.symptom.replace("_", " ")
        return f"Do you have {symptom_text}?"
    
    def _generate_explanation(self) -> str:
        """Згенерувати пояснення"""
        n_yes = len(self.diseases_if_yes)
        n_no = len(self.diseases_if_no)
        
        if n_yes > 0 and n_no > 0:
            return (
                f"This symptom helps distinguish between {n_yes} and {n_no} diagnoses. "
                f"Information gain: {self.information_gain:.3f}"
            )
        return f"Information gain: {self.information_gain:.3f}"


# Симптоми, що можуть вказувати на небезпечні стани
SAFETY_CRITICAL_SYMPTOMS = {
    "chest pain", "chest_pain",
    "shortness of breath", "shortness_of_breath", 
    "difficulty breathing", "difficulty_breathing",
    "loss of consciousness", "loss_of_consciousness",
    "severe headache", "severe_headache",
    "confusion", "altered_sensorium",
    "high fever", "high_fever",
    "seizures", "convulsions",
    "bleeding", "blood in stool", "blood in urine",
    "sudden weakness", "paralysis",
    "severe abdominal pain", "severe_abdominal_pain",
    "suicidal thoughts", "suicidal_ideation",
}


class QuestionGenerator:
    """
    Генератор уточнюючих питань.
    
    Приклад використання:
        generator = QuestionGenerator.from_database(
            "data/unified_disease_symptom_data_full.json"
        )
        
        # Генерація питань
        candidates = ["Influenza", "Common Cold", "Bronchitis"]
        known_symptoms = ["fever", "cough"]
        
        questions = generator.generate(
            candidates=candidates,
            known_present=known_symptoms,
            n_questions=3
        )
        
        for q in questions:
            print(f"Q: {q.text} (IG={q.information_gain:.3f})")
    """
    
    def __init__(
        self,
        disease_symptom_matrix: np.ndarray,
        disease_names: List[str],
        symptom_names: List[str],
        strategy: QuestionStrategy = QuestionStrategy.HYBRID
    ):
        """
        Args:
            disease_symptom_matrix: Матриця (N_diseases, N_symptoms)
            disease_names: Назви діагнозів
            symptom_names: Назви симптомів
            strategy: Стратегія вибору питань
        """
        self.matrix = disease_symptom_matrix
        self.disease_names = disease_names
        self.symptom_names = symptom_names
        self.strategy = strategy
        
        # Калькулятор Information Gain
        self.ig_calculator = InformationGainCalculator(
            disease_symptom_matrix,
            disease_names,
            symptom_names
        )
        
        # Індекси
        self.disease_to_idx = {d: i for i, d in enumerate(disease_names)}
        self.symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}
        
        # Safety-critical симптоми (перетин з наявними)
        self.safety_symptoms = SAFETY_CRITICAL_SYMPTOMS & set(symptom_names)
    
    @classmethod
    def from_database(
        cls,
        database_path: str,
        strategy: QuestionStrategy = QuestionStrategy.HYBRID
    ) -> "QuestionGenerator":
        """
        Створити з бази даних.
        
        Args:
            database_path: Шлях до JSON бази даних
            strategy: Стратегія
            
        Returns:
            QuestionGenerator
        """
        encoder = DiseaseEncoder.from_database(database_path)
        matrix = encoder.encode_all(normalize=False)
        
        return cls(
            disease_symptom_matrix=matrix,
            disease_names=encoder.disease_names,
            symptom_names=encoder.vocabulary.symptoms,
            strategy=strategy
        )
    
    @classmethod
    def from_encoder(
        cls,
        encoder: DiseaseEncoder,
        strategy: QuestionStrategy = QuestionStrategy.HYBRID
    ) -> "QuestionGenerator":
        """
        Створити з DiseaseEncoder.
        
        Args:
            encoder: DiseaseEncoder
            strategy: Стратегія
            
        Returns:
            QuestionGenerator
        """
        matrix = encoder.encode_all(normalize=False)
        
        return cls(
            disease_symptom_matrix=matrix,
            disease_names=encoder.disease_names,
            symptom_names=encoder.vocabulary.symptoms,
            strategy=strategy
        )
    
    def _compute_priority_score(
        self,
        ig_result: SymptomInformationGain,
        strategy: QuestionStrategy
    ) -> float:
        """
        Обчислити пріоритетний score для питання.
        
        Args:
            ig_result: Результат Information Gain
            strategy: Стратегія
            
        Returns:
            Priority score
        """
        ig = ig_result.information_gain
        split = ig_result.split_ratio
        is_safety = ig_result.symptom in self.safety_symptoms
        
        if strategy == QuestionStrategy.MAX_INFO_GAIN:
            return ig
        
        elif strategy == QuestionStrategy.BALANCED_SPLIT:
            # Бонус за збалансоване розділення (близьке до 0.5)
            balance_score = 1.0 - abs(split - 0.5) * 2
            return ig * (0.5 + 0.5 * balance_score)
        
        elif strategy == QuestionStrategy.SAFETY_FIRST:
            # Пріоритет safety-critical симптомам
            safety_bonus = 2.0 if is_safety else 1.0
            return ig * safety_bonus
        
        else:  # HYBRID
            # Комбінація всіх факторів
            balance_score = 1.0 - abs(split - 0.5) * 2
            safety_bonus = 1.5 if is_safety else 1.0
            
            return ig * (0.7 + 0.3 * balance_score) * safety_bonus
    
    def generate(
        self,
        candidates: List[str],
        known_present: Optional[List[str]] = None,
        known_absent: Optional[List[str]] = None,
        disease_scores: Optional[Dict[str, float]] = None,
        n_questions: int = 5,
        strategy: Optional[QuestionStrategy] = None
    ) -> List[Question]:
        """
        Згенерувати питання для уточнення діагнозу.
        
        Args:
            candidates: Список кандидатів
            known_present: Відомі присутні симптоми
            known_absent: Відомі відсутні симптоми
            disease_scores: Scores діагнозів від Neural Network
            n_questions: Кількість питань
            strategy: Override стратегії
            
        Returns:
            Список Question, відсортованих за пріоритетом
        """
        known_present = known_present or []
        known_absent = known_absent or []
        strategy = strategy or self.strategy
        
        # Обчислюємо Information Gain
        ig_results = self.ig_calculator.compute_all(
            candidates=candidates,
            known_present=known_present,
            known_absent=known_absent,
            disease_weights=disease_scores
        )
        
        if not ig_results:
            return []
        
        # Створюємо кандидатів на питання
        question_candidates = []
        
        for ig_result in ig_results:
            priority = self._compute_priority_score(ig_result, strategy)
            
            candidate = QuestionCandidate(
                symptom=ig_result.symptom,
                information_gain=ig_result.information_gain,
                split_ratio=ig_result.split_ratio,
                is_safety_critical=ig_result.symptom in self.safety_symptoms,
                priority_score=priority,
                diseases_if_yes=ig_result.diseases_with_symptom,
                diseases_if_no=ig_result.diseases_without_symptom
            )
            
            question_candidates.append(candidate)
        
        # Сортуємо за priority score
        question_candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Конвертуємо в Question
        questions = []
        for candidate in question_candidates[:n_questions]:
            questions.append(candidate.to_question())
        
        return questions
    
    def generate_single(
        self,
        candidates: List[str],
        known_present: Optional[List[str]] = None,
        known_absent: Optional[List[str]] = None,
        disease_scores: Optional[Dict[str, float]] = None,
        strategy: Optional[QuestionStrategy] = None
    ) -> Optional[Question]:
        """
        Згенерувати одне найкраще питання.
        
        Args:
            candidates: Список кандидатів
            known_present: Відомі присутні симптоми
            known_absent: Відомі відсутні симптоми
            disease_scores: Scores діагнозів
            strategy: Стратегія
            
        Returns:
            Question або None
        """
        questions = self.generate(
            candidates=candidates,
            known_present=known_present,
            known_absent=known_absent,
            disease_scores=disease_scores,
            n_questions=1,
            strategy=strategy
        )
        
        return questions[0] if questions else None
    
    def get_symptom_coverage(
        self,
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Отримати покриття симптомів серед кандидатів.
        
        Args:
            candidates: Список кандидатів
            
        Returns:
            Словник {symptom: coverage_ratio}
        """
        candidate_indices = []
        for disease in candidates:
            if disease in self.disease_to_idx:
                candidate_indices.append(self.disease_to_idx[disease])
        
        if not candidate_indices:
            return {}
        
        candidate_indices = np.array(candidate_indices)
        n_candidates = len(candidate_indices)
        
        coverage = {}
        for symptom_idx, symptom in enumerate(self.symptom_names):
            symptom_vector = self.matrix[candidate_indices, symptom_idx]
            coverage[symptom] = symptom_vector.sum() / n_candidates
        
        return coverage
    
    def get_common_symptoms(
        self,
        candidates: List[str],
        min_coverage: float = 0.5
    ) -> List[str]:
        """
        Отримати симптоми, спільні для більшості кандидатів.
        
        Args:
            candidates: Список кандидатів
            min_coverage: Мінімальне покриття
            
        Returns:
            Список симптомів
        """
        coverage = self.get_symptom_coverage(candidates)
        
        common = [
            symptom for symptom, cov in coverage.items()
            if cov >= min_coverage
        ]
        
        return common
    
    def get_distinguishing_symptoms(
        self,
        disease1: str,
        disease2: str
    ) -> Tuple[List[str], List[str]]:
        """
        Отримати симптоми, що відрізняють два діагнози.
        
        Args:
            disease1: Перший діагноз
            disease2: Другий діагноз
            
        Returns:
            (only_in_disease1, only_in_disease2)
        """
        if disease1 not in self.disease_to_idx or disease2 not in self.disease_to_idx:
            return [], []
        
        idx1 = self.disease_to_idx[disease1]
        idx2 = self.disease_to_idx[disease2]
        
        vec1 = self.matrix[idx1]
        vec2 = self.matrix[idx2]
        
        only_in_1 = []
        only_in_2 = []
        
        for symptom_idx, symptom in enumerate(self.symptom_names):
            if vec1[symptom_idx] > 0 and vec2[symptom_idx] == 0:
                only_in_1.append(symptom)
            elif vec2[symptom_idx] > 0 and vec1[symptom_idx] == 0:
                only_in_2.append(symptom)
        
        return only_in_1, only_in_2
    
    def explain_question(
        self,
        question: Question,
        candidates: List[str]
    ) -> str:
        """
        Пояснити чому питання важливе.
        
        Args:
            question: Питання
            candidates: Кандидати
            
        Returns:
            Текстове пояснення
        """
        ig_result = self.ig_calculator.compute_for_symptom(
            symptom=question.symptom,
            candidate_indices=self.ig_calculator._get_candidate_indices(candidates)
        )
        
        n_yes = len(ig_result.diseases_with_symptom)
        n_no = len(ig_result.diseases_without_symptom)
        
        explanation = []
        explanation.append(f"Question about: {question.symptom}")
        explanation.append(f"Information Gain: {ig_result.information_gain:.4f}")
        explanation.append(f"")
        
        if n_yes > 0:
            explanation.append(f"If YES ({n_yes} diagnoses possible):")
            for d in ig_result.diseases_with_symptom[:5]:
                explanation.append(f"  • {d}")
            if n_yes > 5:
                explanation.append(f"  ... and {n_yes - 5} more")
        
        if n_no > 0:
            explanation.append(f"")
            explanation.append(f"If NO ({n_no} diagnoses possible):")
            for d in ig_result.diseases_without_symptom[:5]:
                explanation.append(f"  • {d}")
            if n_no > 5:
                explanation.append(f"  ... and {n_no - 5} more")
        
        return "\n".join(explanation)
    
    @property
    def n_diseases(self) -> int:
        return len(self.disease_names)
    
    @property
    def n_symptoms(self) -> int:
        return len(self.symptom_names)
    
    def __repr__(self) -> str:
        return f"QuestionGenerator(diseases={self.n_diseases}, symptoms={self.n_symptoms}, strategy={self.strategy.value})"
