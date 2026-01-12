"""
Dr.Case — Метрики якості Neural Network

Метрики:
- Recall@k — частка випадків де правильний діагноз в топ-k
- mAP (Mean Average Precision) — середня точність ранжування
- Hamming Loss — частка неправильних передбачень
- Top-1/5/10 Accuracy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum


class NNQualityLevel(Enum):
    """Рівень якості NN"""
    PRODUCTION = "production"
    PROTOTYPE = "prototype"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class NNQualityThresholds:
    """Пороги якості NN"""
    # Recall@k
    recall_1_production: float = 0.70
    recall_1_prototype: float = 0.50
    
    recall_5_production: float = 0.95
    recall_5_prototype: float = 0.85
    
    recall_10_production: float = 0.99
    recall_10_prototype: float = 0.92
    
    # mAP
    map_production: float = 0.80
    map_prototype: float = 0.60
    
    # Hamming Loss (менше = краще)
    hamming_production: float = 0.05
    hamming_prototype: float = 0.10


@dataclass
class NNQualityReport:
    """Звіт про якість Neural Network"""
    # Recall@k
    recall_1: float
    recall_5: float
    recall_10: float
    
    # mAP
    mean_average_precision: float
    
    # Hamming Loss
    hamming_loss: float
    
    # Рівні
    recall_level: NNQualityLevel
    map_level: NNQualityLevel
    overall_level: NNQualityLevel
    
    # Статистика
    total_cases: int
    avg_predictions_per_case: float
    
    # Додаткові метрики
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    confusion_summary: Dict[str, int] = field(default_factory=dict)
    
    def is_production_ready(self) -> bool:
        """Чи готово до production"""
        return self.overall_level == NNQualityLevel.PRODUCTION
    
    def is_acceptable(self) -> bool:
        """Чи прийнятна якість"""
        return self.overall_level in [
            NNQualityLevel.PRODUCTION,
            NNQualityLevel.PROTOTYPE,
            NNQualityLevel.ACCEPTABLE
        ]
    
    def to_dict(self) -> dict:
        """Серіалізація"""
        return {
            "recall_1": self.recall_1,
            "recall_5": self.recall_5,
            "recall_10": self.recall_10,
            "mean_average_precision": self.mean_average_precision,
            "hamming_loss": self.hamming_loss,
            "recall_level": self.recall_level.value,
            "map_level": self.map_level.value,
            "overall_level": self.overall_level.value,
            "total_cases": self.total_cases,
            "avg_predictions_per_case": self.avg_predictions_per_case,
            "top_k_accuracy": self.top_k_accuracy,
            "is_production_ready": self.is_production_ready(),
            "is_acceptable": self.is_acceptable(),
        }
    
    def __repr__(self) -> str:
        return (
            f"NNQualityReport(\n"
            f"  Recall@1:  {self.recall_1:.4f}\n"
            f"  Recall@5:  {self.recall_5:.4f}\n"
            f"  Recall@10: {self.recall_10:.4f}\n"
            f"  mAP:       {self.mean_average_precision:.4f}\n"
            f"  Hamming:   {self.hamming_loss:.4f}\n"
            f"  Level:     {self.overall_level.value}\n"
            f"  Cases:     {self.total_cases}\n"
            f")"
        )


class NNQualityValidator:
    """
    Валідатор якості Neural Network.
    
    Приклад використання:
        validator = NNQualityValidator()
        
        # Варіант 1: з готовими передбаченнями
        report = validator.validate(
            predictions=[
                {"Influenza": 0.8, "Cold": 0.5, "COVID": 0.3},
                {"Diabetes": 0.9, "Obesity": 0.4},
            ],
            true_labels=["Influenza", "Diabetes"]
        )
        
        # Варіант 2: з моделлю
        report = validator.validate_model(
            model=nn_model,
            test_loader=test_loader,
            disease_names=disease_names
        )
    """
    
    def __init__(self, thresholds: Optional[NNQualityThresholds] = None):
        self.thresholds = thresholds or NNQualityThresholds()
    
    def validate(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str]
    ) -> NNQualityReport:
        """
        Валідація з готовими передбаченнями.
        
        Args:
            predictions: [{disease: probability}, ...]
            true_labels: [true_diagnosis, ...]
            
        Returns:
            NNQualityReport
        """
        if not predictions or not true_labels:
            return self._empty_report()
        
        assert len(predictions) == len(true_labels), \
            "Кількість передбачень має дорівнювати кількості міток"
        
        total = len(predictions)
        
        # Recall@k
        recall_1 = self._compute_recall_at_k(predictions, true_labels, k=1)
        recall_5 = self._compute_recall_at_k(predictions, true_labels, k=5)
        recall_10 = self._compute_recall_at_k(predictions, true_labels, k=10)
        
        # mAP
        mAP = self._compute_map(predictions, true_labels)
        
        # Hamming Loss (для multilabel)
        hamming = self._compute_hamming_loss(predictions, true_labels)
        
        # Top-k accuracy
        top_k_acc = {}
        for k in [1, 3, 5, 10, 20]:
            top_k_acc[k] = self._compute_recall_at_k(predictions, true_labels, k)
        
        # Середня кількість передбачень
        avg_preds = np.mean([len(p) for p in predictions])
        
        # Рівні якості
        recall_level = self._classify_recall(recall_5, recall_10)
        map_level = self._classify_map(mAP)
        overall_level = self._compute_overall(recall_level, map_level)
        
        return NNQualityReport(
            recall_1=recall_1,
            recall_5=recall_5,
            recall_10=recall_10,
            mean_average_precision=mAP,
            hamming_loss=hamming,
            recall_level=recall_level,
            map_level=map_level,
            overall_level=overall_level,
            total_cases=total,
            avg_predictions_per_case=avg_preds,
            top_k_accuracy=top_k_acc,
        )
    
    def validate_model(
        self,
        model: Any,
        test_data: List[Tuple[np.ndarray, np.ndarray, str]],
        disease_names: List[str],
        threshold: float = 0.5
    ) -> NNQualityReport:
        """
        Валідація моделі на тестових даних.
        
        Args:
            model: PyTorch модель
            test_data: [(symptom_vector, som_context, true_diagnosis), ...]
            disease_names: Список назв хвороб
            threshold: Поріг для бінарного передбачення
            
        Returns:
            NNQualityReport
        """
        import torch
        
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for symptom_vec, som_ctx, true_diag in test_data:
                # Конвертуємо в тензори
                x_symptoms = torch.tensor(symptom_vec, dtype=torch.float32).unsqueeze(0)
                x_som = torch.tensor(som_ctx, dtype=torch.float32).unsqueeze(0)
                
                # Передбачення
                outputs = model(x_symptoms, x_som)
                # SOFTMAX (не sigmoid!) - бо CrossEntropyLoss при навчанні
                probs = torch.softmax(outputs, dim=-1).squeeze().numpy()
                
                # Конвертуємо в словник
                pred_dict = {
                    disease_names[i]: float(probs[i])
                    for i in range(len(disease_names))
                }
                
                predictions.append(pred_dict)
                true_labels.append(true_diag)
        
        return self.validate(predictions, true_labels)
    
    def _compute_recall_at_k(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str],
        k: int
    ) -> float:
        """Recall@k — частка випадків де true_label в топ-k"""
        hits = 0
        
        for pred, true_label in zip(predictions, true_labels):
            # Сортуємо за ймовірністю
            sorted_preds = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            top_k_diseases = [d for d, _ in sorted_preds[:k]]
            
            if true_label in top_k_diseases:
                hits += 1
        
        return hits / len(predictions) if predictions else 0.0
    
    def _compute_map(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str]
    ) -> float:
        """Mean Average Precision"""
        aps = []
        
        for pred, true_label in zip(predictions, true_labels):
            # Сортуємо за ймовірністю
            sorted_preds = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            
            # Знаходимо ранг правильного діагнозу
            for rank, (disease, _) in enumerate(sorted_preds, 1):
                if disease == true_label:
                    # AP = 1/rank для single-label
                    aps.append(1.0 / rank)
                    break
            else:
                # Якщо не знайдено
                aps.append(0.0)
        
        return np.mean(aps) if aps else 0.0
    
    def _compute_hamming_loss(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str],
        threshold: float = 0.5
    ) -> float:
        """
        Hamming Loss — частка неправильних бінарних передбачень.
        
        Для single-label: (1 - Recall@1)
        """
        # Спрощена версія для single-label
        return 1.0 - self._compute_recall_at_k(predictions, true_labels, k=1)
    
    def _classify_recall(self, recall_5: float, recall_10: float) -> NNQualityLevel:
        """Класифікація за Recall"""
        if recall_5 >= self.thresholds.recall_5_production and \
           recall_10 >= self.thresholds.recall_10_production:
            return NNQualityLevel.PRODUCTION
        elif recall_5 >= self.thresholds.recall_5_prototype and \
             recall_10 >= self.thresholds.recall_10_prototype:
            return NNQualityLevel.PROTOTYPE
        elif recall_5 >= 0.7:
            return NNQualityLevel.ACCEPTABLE
        else:
            return NNQualityLevel.POOR
    
    def _classify_map(self, mAP: float) -> NNQualityLevel:
        """Класифікація за mAP"""
        if mAP >= self.thresholds.map_production:
            return NNQualityLevel.PRODUCTION
        elif mAP >= self.thresholds.map_prototype:
            return NNQualityLevel.PROTOTYPE
        elif mAP >= 0.4:
            return NNQualityLevel.ACCEPTABLE
        else:
            return NNQualityLevel.POOR
    
    def _compute_overall(
        self,
        recall_level: NNQualityLevel,
        map_level: NNQualityLevel
    ) -> NNQualityLevel:
        """Загальний рівень"""
        levels = [recall_level, map_level]
        
        if NNQualityLevel.POOR in levels:
            return NNQualityLevel.POOR
        
        if all(l == NNQualityLevel.PRODUCTION for l in levels):
            return NNQualityLevel.PRODUCTION
        elif NNQualityLevel.PRODUCTION in levels or \
             all(l in [NNQualityLevel.PRODUCTION, NNQualityLevel.PROTOTYPE] for l in levels):
            return NNQualityLevel.PROTOTYPE
        else:
            return NNQualityLevel.ACCEPTABLE
    
    def _empty_report(self) -> NNQualityReport:
        """Порожній звіт"""
        return NNQualityReport(
            recall_1=0.0,
            recall_5=0.0,
            recall_10=0.0,
            mean_average_precision=0.0,
            hamming_loss=1.0,
            recall_level=NNQualityLevel.POOR,
            map_level=NNQualityLevel.POOR,
            overall_level=NNQualityLevel.POOR,
            total_cases=0,
            avg_predictions_per_case=0.0,
        )
    
    def validate_from_checkpoint(
        self,
        nn_checkpoint_path: str,
        som_checkpoint_path: str,
        database_path: str,
        n_samples: int = 1000,
        dropout_rate: float = 0.3
    ) -> NNQualityReport:
        """
        Валідація з checkpoint файлів.
        
        Args:
            nn_checkpoint_path: Шлях до NN checkpoint (.pt)
            som_checkpoint_path: Шлях до SOM checkpoint (.pkl)
            database_path: Шлях до бази хвороб
            n_samples: Кількість тестових випадків
            dropout_rate: Частка пропущених симптомів
            
        Returns:
            NNQualityReport
        """
        import torch
        import pickle
        import json
        
        # Завантажуємо NN
        nn_checkpoint = torch.load(nn_checkpoint_path, map_location='cpu', weights_only=False)
        
        from dr_case.neural_network.two_branch_model import TwoBranchNN
        
        model_config = nn_checkpoint.get('model_config', {})
        n_symptoms = model_config.get('n_symptoms', 460)
        n_diseases = model_config.get('n_diseases', 844)
        som_dim = model_config.get('som_dim', 10)
        
        model = TwoBranchNN(
            n_symptoms=n_symptoms,
            n_diseases=n_diseases,
            som_dim=som_dim
        )
        model.load_state_dict(nn_checkpoint.get('model_state'))
        model.eval()
        
        disease_names = nn_checkpoint.get('disease_names', [])
        
        # Завантажуємо SOM
        with open(som_checkpoint_path, 'rb') as f:
            som_data = pickle.load(f)
        
        som_model = som_data.get('som')
        
        # Завантажуємо базу
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        # Створюємо словник симптомів (lowercase для відповідності NN)
        symptom_names = nn_checkpoint.get('symptom_names', [])
        if not symptom_names:
            # Fallback: з бази даних
            all_symptoms = set()
            for disease_data in database.values():
                all_symptoms.update(disease_data.get('symptoms', []))
            symptom_names = sorted([s.strip().lower() for s in all_symptoms])
        
        # Генеруємо тестові дані
        test_data = []
        diseases = list(database.keys())
        
        np.random.seed(42)
        for _ in range(n_samples):
            disease = np.random.choice(diseases)
            symptoms = database[disease].get('symptoms', [])
            
            if not symptoms:
                continue
            
            # Dropout
            n_keep = max(1, int(len(symptoms) * (1 - dropout_rate)))
            kept_symptoms = np.random.choice(
                symptoms, size=min(n_keep, len(symptoms)), replace=False
            )
            
            # Symptom vector (lowercase для відповідності NN)
            symptom_vector = np.zeros(n_symptoms, dtype=np.float32)
            for s in kept_symptoms:
                s_lower = s.strip().lower()
                if s_lower in symptom_names:
                    idx = symptom_names.index(s_lower)
                    if idx < n_symptoms:
                        symptom_vector[idx] = 1.0
            
            # L2 нормалізація (модель очікує нормалізовані дані)
            norm = np.linalg.norm(symptom_vector)
            if norm > 0:
                symptom_vector = symptom_vector / norm
            
            # SOM context (повний softmax membership)
            if som_model:
                h, w = som_model._weights.shape[:2]
                distances = []
                for i in range(h):
                    for j in range(w):
                        w_vec = som_model._weights[i, j]
                        dist = np.linalg.norm(symptom_vector - w_vec)
                        distances.append(dist)
                
                sorted_idx = np.argsort(distances)[:som_dim]
                top_dists = np.array([distances[i] for i in sorted_idx])
                
                # Softmax membership
                exp_neg = np.exp(-top_dists)
                som_context = (exp_neg / exp_neg.sum()).astype(np.float32)
            else:
                som_context = np.zeros(som_dim, dtype=np.float32)
            
            test_data.append((symptom_vector, som_context, disease))
        
        return self.validate_model(
            model=model,
            test_data=test_data,
            disease_names=disease_names
        )
