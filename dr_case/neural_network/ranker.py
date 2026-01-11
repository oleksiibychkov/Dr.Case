"""
Dr.Case — Ранжування діагнозів

DiagnosisRanker використовує навчену Neural Network
для ранжування кандидатів від Candidate Selector.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from dr_case.schemas import DiagnosisHypothesis, CaseRecord
from dr_case.encoding import PatientEncoder, SymptomVocabulary
from .trainer import NNTrainer


@dataclass
class RankingResult:
    """Результат ранжування"""
    # Ранжований список діагнозів
    ranked_diseases: List[str]
    scores: List[float]
    probabilities: List[float]
    
    # Симптоми
    present_symptoms: List[str]
    absent_symptoms: List[str]
    
    @property
    def top_disease(self) -> str:
        """Топ діагноз"""
        return self.ranked_diseases[0] if self.ranked_diseases else ""
    
    @property
    def top_score(self) -> float:
        """Score топ діагнозу"""
        return self.scores[0] if self.scores else 0.0
    
    @property
    def top_probability(self) -> float:
        """Ймовірність топ діагнозу"""
        return self.probabilities[0] if self.probabilities else 0.0
    
    def get_top_n(self, n: int = 5) -> List[Tuple[str, float, float]]:
        """Отримати топ-N: (disease, score, probability)"""
        return list(zip(
            self.ranked_diseases[:n],
            self.scores[:n],
            self.probabilities[:n]
        ))
    
    def to_hypotheses(self, n: int = 10) -> List[DiagnosisHypothesis]:
        """Конвертувати в список DiagnosisHypothesis"""
        hypotheses = []
        
        for disease, score, prob in self.get_top_n(n):
            hypothesis = DiagnosisHypothesis(
                disease_name=disease,
                confidence=prob,
                matching_symptoms=self.present_symptoms.copy()
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def contains(self, disease_name: str) -> bool:
        """Перевірити чи діагноз є в ранжуванні"""
        return disease_name in self.ranked_diseases
    
    def get_rank(self, disease_name: str) -> int:
        """Отримати ранг діагнозу (1-based)"""
        try:
            return self.ranked_diseases.index(disease_name) + 1
        except ValueError:
            return -1


class DiagnosisRanker:
    """
    Ранжування діагнозів-кандидатів.
    
    Приклад використання:
        # Завантаження
        ranker = DiagnosisRanker.load(
            model_path="models/nn_model.pt",
            database_path="data/unified_disease_symptom_data_full.json"
        )
        
        # Ранжування
        candidates = ["Influenza", "Common Cold", "COVID-19", "Bronchitis"]
        symptoms = ["fever", "cough", "headache"]
        
        result = ranker.rank(symptoms, candidates)
        
        print(f"Top diagnosis: {result.top_disease}")
        print(f"Confidence: {result.top_probability:.2%}")
    """
    
    def __init__(
        self,
        trainer: NNTrainer,
        vocabulary: SymptomVocabulary
    ):
        """
        Args:
            trainer: Навчений NNTrainer
            vocabulary: Словник симптомів
        """
        self.trainer = trainer
        self.vocabulary = vocabulary
        self.patient_encoder = PatientEncoder(vocabulary)
        
        # Маппінг
        self.disease_to_idx = trainer.disease_to_idx
        self.idx_to_disease = {v: k for k, v in self.disease_to_idx.items()}
    
    @classmethod
    def load(
        cls,
        model_path: str,
        database_path: Optional[str] = None,
        device: Optional[str] = None
    ) -> "DiagnosisRanker":
        """
        Завантажити з файлів.
        
        Args:
            model_path: Шлях до .pt файлу моделі
            database_path: Шлях до бази даних
            device: Device для моделі
            
        Returns:
            DiagnosisRanker
        """
        trainer = NNTrainer.load(model_path, device)
        
        # Vocabulary
        if database_path is None:
            model_dir = Path(model_path).parent
            database_path = model_dir.parent / "data" / "unified_disease_symptom_data_full.json"
        
        vocabulary = SymptomVocabulary.from_database(str(database_path))
        
        return cls(trainer, vocabulary)
    
    def rank(
        self,
        present_symptoms: List[str],
        candidates: List[str],
        absent_symptoms: Optional[List[str]] = None
    ) -> RankingResult:
        """
        Ранжувати кандидатів за симптомами.
        
        Args:
            present_symptoms: Присутні симптоми
            candidates: Список кандидатів для ранжування
            absent_symptoms: Відсутні симптоми
            
        Returns:
            RankingResult
        """
        absent_symptoms = absent_symptoms or []
        
        # Кодуємо симптоми
        if absent_symptoms:
            symptom_vector = self.patient_encoder.encode_with_negatives(
                present=present_symptoms,
                absent=absent_symptoms
            )
        else:
            symptom_vector = self.patient_encoder.encode(present_symptoms)
        
        # Отримуємо індекси кандидатів
        candidate_indices = []
        valid_candidates = []
        
        for disease in candidates:
            if disease in self.disease_to_idx:
                candidate_indices.append(self.disease_to_idx[disease])
                valid_candidates.append(disease)
        
        if not candidate_indices:
            return RankingResult(
                ranked_diseases=[],
                scores=[],
                probabilities=[],
                present_symptoms=present_symptoms,
                absent_symptoms=absent_symptoms
            )
        
        # Передбачення
        scores, indices = self.trainer.predict(symptom_vector, candidate_indices)
        
        # Фільтруємо тільки кандидатів
        ranked_diseases = []
        filtered_scores = []
        
        for idx, score in zip(indices, scores):
            if idx in candidate_indices:
                ranked_diseases.append(self.idx_to_disease[idx])
                filtered_scores.append(score)
        
        # Обчислюємо ймовірності (softmax)
        if filtered_scores:
            scores_arr = np.array(filtered_scores)
            exp_scores = np.exp(scores_arr - np.max(scores_arr))  # стабільність
            probabilities = exp_scores / exp_scores.sum()
        else:
            probabilities = []
        
        return RankingResult(
            ranked_diseases=ranked_diseases,
            scores=filtered_scores,
            probabilities=probabilities.tolist() if len(probabilities) > 0 else [],
            present_symptoms=present_symptoms,
            absent_symptoms=absent_symptoms
        )
    
    def rank_from_case(
        self,
        case: CaseRecord,
        candidates: List[str]
    ) -> RankingResult:
        """
        Ранжувати кандидатів з CaseRecord.
        
        Args:
            case: Клінічний випадок
            candidates: Список кандидатів
            
        Returns:
            RankingResult
        """
        return self.rank(
            present_symptoms=case.present_symptom_names,
            candidates=candidates,
            absent_symptoms=case.absent_symptom_names
        )
    
    def rank_from_vector(
        self,
        symptom_vector: np.ndarray,
        candidates: List[str]
    ) -> RankingResult:
        """
        Ранжувати за вектором симптомів.
        
        Args:
            symptom_vector: Вектор симптомів
            candidates: Список кандидатів
            
        Returns:
            RankingResult
        """
        # Отримуємо індекси кандидатів
        candidate_indices = []
        valid_candidates = []
        
        for disease in candidates:
            if disease in self.disease_to_idx:
                candidate_indices.append(self.disease_to_idx[disease])
                valid_candidates.append(disease)
        
        # Передбачення
        scores, indices = self.trainer.predict(symptom_vector, candidate_indices)
        
        # Фільтруємо
        ranked_diseases = []
        filtered_scores = []
        
        for idx, score in zip(indices, scores):
            if idx in candidate_indices:
                ranked_diseases.append(self.idx_to_disease[idx])
                filtered_scores.append(score)
        
        # Ймовірності
        if filtered_scores:
            scores_arr = np.array(filtered_scores)
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            probabilities = exp_scores / exp_scores.sum()
        else:
            probabilities = []
        
        # Декодуємо симптоми
        present, absent = self.patient_encoder.decode(symptom_vector)
        
        return RankingResult(
            ranked_diseases=ranked_diseases,
            scores=filtered_scores,
            probabilities=probabilities.tolist() if len(probabilities) > 0 else [],
            present_symptoms=present,
            absent_symptoms=absent
        )
    
    def rank_all(
        self,
        present_symptoms: List[str],
        absent_symptoms: Optional[List[str]] = None,
        top_n: int = 10
    ) -> RankingResult:
        """
        Ранжувати ВСІ діагнози (без обмеження кандидатами).
        
        Args:
            present_symptoms: Симптоми
            absent_symptoms: Відсутні симптоми
            top_n: Кількість топ результатів
            
        Returns:
            RankingResult з топ-N діагнозами
        """
        absent_symptoms = absent_symptoms or []
        
        # Кодуємо
        if absent_symptoms:
            symptom_vector = self.patient_encoder.encode_with_negatives(
                present=present_symptoms,
                absent=absent_symptoms
            )
        else:
            symptom_vector = self.patient_encoder.encode(present_symptoms)
        
        # Передбачення без маски
        scores, indices = self.trainer.predict(symptom_vector)
        
        # Топ-N
        ranked_diseases = [self.idx_to_disease[idx] for idx in indices[:top_n]]
        filtered_scores = scores[:top_n].tolist()
        
        # Ймовірності
        scores_arr = np.array(filtered_scores)
        exp_scores = np.exp(scores_arr - np.max(scores_arr))
        probabilities = exp_scores / exp_scores.sum()
        
        return RankingResult(
            ranked_diseases=ranked_diseases,
            scores=filtered_scores,
            probabilities=probabilities.tolist(),
            present_symptoms=present_symptoms,
            absent_symptoms=absent_symptoms
        )
    
    def evaluate_accuracy(
        self,
        test_symptoms: List[List[str]],
        true_diseases: List[str],
        candidates_lists: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Оцінити точність ранжування.
        
        Args:
            test_symptoms: Списки симптомів
            true_diseases: Правильні діагнози
            candidates_lists: Списки кандидатів (опціонально)
            
        Returns:
            Метрики
        """
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        mrr_sum = 0  # Mean Reciprocal Rank
        
        for i, (symptoms, true_disease) in enumerate(zip(test_symptoms, true_diseases)):
            if candidates_lists:
                candidates = candidates_lists[i]
                result = self.rank(symptoms, candidates)
            else:
                result = self.rank_all(symptoms, top_n=100)
            
            rank = result.get_rank(true_disease)
            
            if rank == 1:
                top1_correct += 1
            if 1 <= rank <= 5:
                top5_correct += 1
            if 1 <= rank <= 10:
                top10_correct += 1
            
            if rank > 0:
                mrr_sum += 1.0 / rank
        
        n = len(true_diseases)
        
        return {
            "top1_accuracy": top1_correct / n,
            "top5_accuracy": top5_correct / n,
            "top10_accuracy": top10_correct / n,
            "mrr": mrr_sum / n,  # Mean Reciprocal Rank
        }
    
    @property
    def n_diseases(self) -> int:
        """Кількість діагнозів"""
        return len(self.disease_to_idx)
    
    @property
    def n_symptoms(self) -> int:
        """Кількість симптомів"""
        return self.vocabulary.size
    
    def __repr__(self) -> str:
        return f"DiagnosisRanker(diseases={self.n_diseases}, symptoms={self.n_symptoms})"
