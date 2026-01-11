"""
Dr.Case — Оптимізація параметрів SOM

Використовує Optuna для автоматичного підбору оптимальних параметрів:
- Розмір карти (grid_height, grid_width)
- Параметри навчання (epochs, sigma, learning_rate)
- Параметри Candidate Selector (alpha, k, tau)

Цільова функція: максимізувати Candidate Recall при розумній кількості кандидатів.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import numpy as np

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

from dr_case.config import SOMConfig, CandidateSelectorConfig
from dr_case.encoding import DiseaseEncoder
from dr_case.som import SOMModel, SOMProjector


@dataclass
class TuningResult:
    """Результат оптимізації"""
    # Найкращі параметри
    best_params: Dict[str, Any]
    best_value: float
    
    # Метрики найкращої моделі
    best_metrics: Dict[str, float]
    
    # Історія всіх trials
    n_trials: int
    study_name: str
    
    # Конфігурації
    som_config: Optional[SOMConfig] = None
    selector_config: Optional[CandidateSelectorConfig] = None
    
    def to_dict(self) -> Dict:
        """Конвертувати в словник"""
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_metrics": self.best_metrics,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
        }
    
    def save(self, path: str) -> None:
        """Зберегти результат в JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TuningResult":
        """Завантажити результат з JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class SOMTuner:
    """
    Оптимізатор параметрів SOM через Optuna.
    
    Приклад використання:
        tuner = SOMTuner(database_path="data/unified_disease_symptom_data_full.json")
        
        # Запустити оптимізацію
        result = tuner.optimize(n_trials=50)
        
        print(f"Best Recall: {result.best_metrics['recall']:.2%}")
        print(f"Best params: {result.best_params}")
        
        # Отримати оптимальну модель
        som, projector = tuner.get_best_model()
    """
    
    def __init__(
        self,
        database_path: str,
        target_recall: float = 0.995,
        max_candidates_ratio: float = 0.15,
        validation_split: float = 0.2,
        random_seed: int = 42
    ):
        """
        Args:
            database_path: Шлях до JSON бази даних
            target_recall: Цільовий Candidate Recall (за замовчуванням 99.5%)
            max_candidates_ratio: Максимальна частка кандидатів від загальної кількості
            validation_split: Частка даних для валідації
            random_seed: Seed для відтворюваності
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        self.database_path = database_path
        self.target_recall = target_recall
        self.max_candidates_ratio = max_candidates_ratio
        self.validation_split = validation_split
        self.random_seed = random_seed
        
        # Завантажуємо дані
        self._load_data()
        
        # Зберігаємо найкращу модель
        self._best_som: Optional[SOMModel] = None
        self._best_projector: Optional[SOMProjector] = None
        self._best_params: Optional[Dict] = None
    
    def _load_data(self) -> None:
        """Завантажити та підготувати дані"""
        print(f"Loading data from {self.database_path}...")
        
        self.encoder = DiseaseEncoder.from_database(self.database_path)
        self.disease_matrix = self.encoder.encode_all(normalize=False)
        self.disease_names = self.encoder.disease_names
        
        self.n_diseases = len(self.disease_names)
        self.n_symptoms = self.encoder.vector_dim
        
        print(f"  Diseases: {self.n_diseases}")
        print(f"  Symptoms: {self.n_symptoms}")
        
        # Розділяємо на train/validation
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.n_diseases)
        
        val_size = int(self.n_diseases * self.validation_split)
        self.val_indices = indices[:val_size]
        self.train_indices = indices[val_size:]
        
        print(f"  Train: {len(self.train_indices)}, Validation: {len(self.val_indices)}")
    
    def _create_objective(self) -> callable:
        """Створити objective функцію для Optuna"""
        
        def objective(trial: Trial) -> float:
            # 1. Параметри SOM
            grid_size = trial.suggest_int("grid_size", 10, 30)
            epochs = trial.suggest_int("epochs", 200, 1000, step=100)
            sigma_ratio = trial.suggest_float("sigma_ratio", 0.3, 0.7)
            lr_init = trial.suggest_float("lr_init", 0.1, 0.8)
            lr_final = trial.suggest_float("lr_final", 0.001, 0.05)
            
            # 2. Параметри Candidate Selector
            alpha = trial.suggest_float("alpha", 0.85, 0.99)
            k = trial.suggest_int("k", 4, 15)
            tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
            lambda_param = trial.suggest_float("lambda_param", 0.5, 3.0)
            
            # Створюємо конфігурацію SOM
            som_config = SOMConfig(
                grid_height=grid_size,
                grid_width=grid_size,
                input_dim=self.n_symptoms,
                epochs=epochs,
                sigma_init=grid_size * sigma_ratio,
                learning_rate_init=lr_init
            )
            
            # Навчаємо SOM
            try:
                som = SOMModel(som_config)
                som.train(
                    data=self.disease_matrix,
                    disease_names=self.disease_names,
                    epochs=epochs,
                    verbose=False
                )
            except Exception as e:
                print(f"  Training failed: {e}")
                return 0.0  # Мінімальне значення
            
            # Створюємо Projector
            selector_config = CandidateSelectorConfig(
                alpha=alpha,
                k=k,
                tau=tau,
                membership_lambda=lambda_param
            )
            projector = SOMProjector(som, selector_config)
            
            # Оцінюємо на валідаційній вибірці
            val_vectors = self.disease_matrix[self.val_indices]
            val_diseases = [self.disease_names[i] for i in self.val_indices]
            
            metrics = projector.evaluate_recall(val_vectors, val_diseases)
            
            recall = metrics["recall"]
            avg_candidates = metrics["avg_candidates"]
            candidates_ratio = avg_candidates / self.n_diseases
            
            # Обчислюємо score
            # Хочемо: високий recall + мала кількість кандидатів
            
            if recall < self.target_recall * 0.9:
                # Якщо recall занадто низький - штраф
                score = recall * 0.5
            elif candidates_ratio > self.max_candidates_ratio:
                # Якщо забагато кандидатів - штраф
                penalty = (candidates_ratio - self.max_candidates_ratio) * 2
                score = recall - penalty
            else:
                # Ідеальний випадок: високий recall, мало кандидатів
                # Бонус за менше кандидатів
                efficiency_bonus = (self.max_candidates_ratio - candidates_ratio) * 0.1
                score = recall + efficiency_bonus
            
            # Зберігаємо метрики
            trial.set_user_attr("recall", recall)
            trial.set_user_attr("avg_candidates", avg_candidates)
            trial.set_user_attr("candidates_ratio", candidates_ratio)
            trial.set_user_attr("qe", som.quantization_error)
            trial.set_user_attr("te", som.topographic_error)
            trial.set_user_attr("fill_ratio", len(som.filled_units) / (grid_size ** 2))
            
            # Зберігаємо найкращу модель
            if not hasattr(self, '_best_score') or score > self._best_score:
                self._best_score = score
                self._best_som = som
                self._best_projector = projector
                self._best_params = trial.params.copy()
            
            return score
        
        return objective
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: str = "som_optimization",
        show_progress: bool = True
    ) -> TuningResult:
        """
        Запустити оптимізацію.
        
        Args:
            n_trials: Кількість trials
            timeout: Ліміт часу в секундах
            study_name: Назва study для Optuna
            show_progress: Показувати прогрес
            
        Returns:
            TuningResult з найкращими параметрами
        """
        print(f"\nStarting optimization: {n_trials} trials")
        print(f"  Target recall: {self.target_recall:.1%}")
        print(f"  Max candidates ratio: {self.max_candidates_ratio:.1%}")
        
        # Створюємо study
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler
        )
        
        # Callback для прогресу
        def progress_callback(study, trial):
            if show_progress and trial.number % 5 == 0:
                best = study.best_trial
                print(f"  Trial {trial.number}: score={trial.value:.4f}, "
                      f"recall={trial.user_attrs.get('recall', 0):.2%}, "
                      f"candidates={trial.user_attrs.get('avg_candidates', 0):.1f}")
        
        # Запускаємо оптимізацію
        self._best_score = -float('inf')
        
        study.optimize(
            self._create_objective(),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[progress_callback] if show_progress else None,
            show_progress_bar=show_progress
        )
        
        # Формуємо результат
        best_trial = study.best_trial
        
        best_metrics = {
            "recall": best_trial.user_attrs.get("recall", 0),
            "avg_candidates": best_trial.user_attrs.get("avg_candidates", 0),
            "candidates_ratio": best_trial.user_attrs.get("candidates_ratio", 0),
            "qe": best_trial.user_attrs.get("qe", 0),
            "te": best_trial.user_attrs.get("te", 0),
            "fill_ratio": best_trial.user_attrs.get("fill_ratio", 0),
        }
        
        # Створюємо конфігурації
        bp = best_trial.params
        
        som_config = SOMConfig(
            grid_height=bp["grid_size"],
            grid_width=bp["grid_size"],
            input_dim=self.n_symptoms,
            epochs=bp["epochs"],
            sigma_init=bp["grid_size"] * bp["sigma_ratio"],
            learning_rate_init=bp["lr_init"]
        )
        
        selector_config = CandidateSelectorConfig(
            alpha=bp["alpha"],
            k=bp["k"],
            tau=bp["tau"],
            membership_lambda=bp["lambda_param"]
        )
        
        result = TuningResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_metrics=best_metrics,
            n_trials=len(study.trials),
            study_name=study_name,
            som_config=som_config,
            selector_config=selector_config
        )
        
        print(f"\n✓ Optimization complete!")
        print(f"  Best score: {result.best_value:.4f}")
        print(f"  Best recall: {best_metrics['recall']:.2%}")
        print(f"  Best avg candidates: {best_metrics['avg_candidates']:.1f}")
        print(f"  Best grid: {bp['grid_size']}x{bp['grid_size']}")
        
        return result
    
    def get_best_model(self) -> Tuple[SOMModel, SOMProjector]:
        """
        Отримати найкращу модель після оптимізації.
        
        Returns:
            (SOMModel, SOMProjector)
        """
        if self._best_som is None:
            raise RuntimeError("No optimization run yet. Call optimize() first.")
        
        return self._best_som, self._best_projector
    
    def train_with_best_params(
        self,
        params: Optional[Dict] = None,
        full_data: bool = True
    ) -> Tuple[SOMModel, SOMProjector, Dict[str, float]]:
        """
        Навчити модель з найкращими параметрами на повних даних.
        
        Args:
            params: Параметри (якщо None - використовує best_params)
            full_data: Чи використовувати повні дані (без train/val split)
            
        Returns:
            (SOMModel, SOMProjector, metrics)
        """
        if params is None:
            if self._best_params is None:
                raise RuntimeError("No best params. Call optimize() first.")
            params = self._best_params
        
        print(f"\nTraining final model with best params...")
        
        # Конфігурація SOM
        som_config = SOMConfig(
            grid_height=params["grid_size"],
            grid_width=params["grid_size"],
            input_dim=self.n_symptoms,
            epochs=params["epochs"],
            sigma_init=params["grid_size"] * params["sigma_ratio"],
            learning_rate_init=params["lr_init"]
        )
        
        # Навчаємо
        som = SOMModel(som_config)
        metrics = som.train(
            data=self.disease_matrix,
            disease_names=self.disease_names,
            epochs=params["epochs"],
            verbose=True
        )
        
        # Projector
        selector_config = CandidateSelectorConfig(
            alpha=params["alpha"],
            k=params["k"],
            tau=params["tau"],
            membership_lambda=params["lambda_param"]
        )
        projector = SOMProjector(som, selector_config)
        
        # Оцінюємо recall
        recall_metrics = projector.evaluate_recall(
            self.disease_matrix,
            self.disease_names
        )
        
        metrics.update(recall_metrics)
        
        print(f"  Final recall: {metrics['recall']:.2%}")
        print(f"  Final avg candidates: {metrics['avg_candidates']:.1f}")
        
        return som, projector, metrics
    
    def quick_tune(self, n_trials: int = 20) -> TuningResult:
        """
        Швидка оптимізація з меншою кількістю trials.
        
        Args:
            n_trials: Кількість trials
            
        Returns:
            TuningResult
        """
        return self.optimize(n_trials=n_trials, study_name="som_quick_tune")
    
    def __repr__(self) -> str:
        return f"SOMTuner(diseases={self.n_diseases}, symptoms={self.n_symptoms})"
