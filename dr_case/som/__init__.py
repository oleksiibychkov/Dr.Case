"""
Dr.Case — Модуль Self-Organizing Map (SOM)

SOM кластеризує діагнози на 2D-карті для швидкого звуження
простору пошуку при діагностиці.

Компоненти:
- SOMModel: Основна модель SOM (обгортка над MiniSom)
- SOMProjector: Проєкція пацієнта на SOM + membership
- SOMTrainer: Зручний інтерфейс для навчання

Приклад використання:
    from dr_case.som import SOMTrainer, SOMModel, SOMProjector
    
    # Навчання з бази даних
    trainer = SOMTrainer()
    som, metrics = trainer.train_from_database("data/unified_disease_symptom_data_full.json")
    
    print(f"QE: {metrics['qe']:.4f}")
    print(f"TE: {metrics['te']:.4f}")
    print(f"Fill ratio: {metrics['fill_ratio']:.2%}")
    
    # Оцінка Candidate Recall
    recall = trainer.evaluate_candidate_recall(alpha=0.9, k=6, tau=0.01)
    print(f"Candidate Recall: {recall['recall']:.2%}")
    
    # Проєкція пацієнта
    projector = trainer.get_projector()
    result = projector.project(patient_vector)
    
    print(f"BMU: {result.bmu}")
    print(f"Candidates: {result.candidate_diseases[:5]}")
    
    # Збереження/завантаження
    trainer.save_model("models/som_model.pkl")
    som = SOMModel.load("models/som_model.pkl")
"""

from .model import SOMModel
from .projector import SOMProjector, ProjectionResult
from .trainer import SOMTrainer


__all__ = [
    "SOMModel",
    "SOMProjector",
    "ProjectionResult",
    "SOMTrainer",
]
