"""
Dr.Case — Модуль нейронної мережі (Neural Network)

Нейронна мережа для ранжування діагнозів-кандидатів.

Компоненти:
- DiagnosisNN: Архітектура мережі (PyTorch)
- NNTrainer: Навчання моделі
- DiagnosisRanker: Інференс (ранжування кандидатів)

Приклад використання:
    from dr_case.neural_network import NNTrainer, DiagnosisRanker
    from dr_case.encoding import DiseaseEncoder
    
    # === НАВЧАННЯ ===
    encoder = DiseaseEncoder.from_database("data/unified_disease_symptom_data_full.json")
    disease_matrix = encoder.encode_all()
    
    trainer = NNTrainer(n_symptoms=461, n_diseases=842)
    history = trainer.train(disease_matrix, encoder.disease_names)
    trainer.save("models/nn_model.pt")
    
    # === ІНФЕРЕНС ===
    ranker = DiagnosisRanker.load("models/nn_model.pt")
    
    candidates = ["Influenza", "Common Cold", "Bronchitis"]
    symptoms = ["fever", "cough", "headache"]
    
    result = ranker.rank(symptoms, candidates)
    
    print(f"Top diagnosis: {result.top_disease}")
    print(f"Confidence: {result.top_probability:.2%}")
    
    for disease, score, prob in result.get_top_n(5):
        print(f"  {disease}: {prob:.2%}")
"""

from .model import DiagnosisNN, DiagnosisNNWithEmbedding, FocalLoss, create_model
from .trainer import NNTrainer, TrainingConfig, DiseaseDataset
from .ranker import DiagnosisRanker, RankingResult


__all__ = [
    # Model
    "DiagnosisNN",
    "DiagnosisNNWithEmbedding",
    "FocalLoss",
    "create_model",
    
    # Trainer
    "NNTrainer",
    "TrainingConfig",
    "DiseaseDataset",
    
    # Ranker
    "DiagnosisRanker",
    "RankingResult",
]
