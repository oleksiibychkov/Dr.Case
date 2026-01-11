"""
Dr.Case — Модуль оптимізації (optimization)

Автоматичний підбір оптимальних параметрів через Optuna.

Компоненти:
- SOMTuner: Оптимізація параметрів SOM та Candidate Selector
- NNTuner: Оптимізація параметрів Neural Network

Приклад використання:
    from dr_case.optimization import SOMTuner, NNTuner
    
    # === SOM Tuner ===
    som_tuner = SOMTuner(
        database_path="data/unified_disease_symptom_data_full.json",
        target_recall=0.995
    )
    som_result = som_tuner.optimize(n_trials=50)
    som, projector = som_tuner.get_best_model()
    
    # === NN Tuner ===
    nn_tuner = NNTuner(
        database_path="data/unified_disease_symptom_data_full.json",
        target_recall_at_5=0.95
    )
    nn_result = nn_tuner.optimize(n_trials=50)
    model = nn_tuner.get_best_model()
    
    # Навчити фінальну модель з найкращими параметрами
    final_model, metrics = nn_tuner.train_with_best_params()
"""

from .som_tuner import SOMTuner, TuningResult
from .nn_tuner import NNTuner, NNTuningResult


__all__ = [
    "SOMTuner",
    "TuningResult",
    "NNTuner",
    "NNTuningResult",
]
