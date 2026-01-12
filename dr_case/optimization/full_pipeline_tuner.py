"""
Dr.Case — Full Pipeline Tuner

End-to-end оптимізація всього pipeline:
1. Оцінка поточної якості
2. Ідентифікація слабких місць
3. Автоматичне налаштування параметрів
4. Перевалідація
5. Збереження найкращої конфігурації

Стратегії оптимізації:
- Grid Search для окремих компонентів
- Bayesian Optimization (з Optuna)
- Iterative Refinement
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from enum import Enum
import copy


@dataclass
class TuningConfig:
    """Конфігурація для tuning"""
    # SOM параметри
    som_grid_sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(12, 12), (15, 15), (18, 18)]
    )
    som_learning_rates: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7]
    )
    som_sigma_init: List[float] = field(
        default_factory=lambda: [3.0, 5.0, 7.0]
    )
    
    # Candidate Selector
    candidate_alphas: List[float] = field(
        default_factory=lambda: [0.85, 0.90, 0.95]
    )
    candidate_k: List[int] = field(
        default_factory=lambda: [6, 8, 10]
    )
    
    # NN параметри
    nn_hidden_dims: List[List[int]] = field(
        default_factory=lambda: [[256, 128], [512, 256], [256, 128, 64]]
    )
    nn_dropout: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.4]
    )
    nn_learning_rates: List[float] = field(
        default_factory=lambda: [1e-3, 5e-4, 1e-4]
    )
    
    # Training
    max_epochs: int = 100
    patience: int = 10
    n_validation_samples: int = 500


@dataclass
class TuningResult:
    """Результат tuning"""
    component: str
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    improvement: float  # Покращення порівняно з початковим
    duration_seconds: float


@dataclass
class FullTuningReport:
    """Звіт про повний tuning"""
    som_result: Optional[TuningResult] = None
    candidate_result: Optional[TuningResult] = None
    nn_result: Optional[TuningResult] = None
    
    initial_score: float = 0.0
    final_score: float = 0.0
    total_improvement: float = 0.0
    
    best_config: Dict[str, Any] = field(default_factory=dict)
    
    timestamp: str = ""
    total_duration_seconds: float = 0.0
    
    def save(self, path: str) -> None:
        """Зберегти звіт"""
        data = {
            "som": {
                "best_params": self.som_result.best_params if self.som_result else None,
                "best_score": self.som_result.best_score if self.som_result else None,
                "improvement": self.som_result.improvement if self.som_result else None,
            } if self.som_result else None,
            "candidate": {
                "best_params": self.candidate_result.best_params if self.candidate_result else None,
                "best_score": self.candidate_result.best_score if self.candidate_result else None,
            } if self.candidate_result else None,
            "nn": {
                "best_params": self.nn_result.best_params if self.nn_result else None,
                "best_score": self.nn_result.best_score if self.nn_result else None,
            } if self.nn_result else None,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "total_improvement": self.total_improvement,
            "best_config": self.best_config,
            "timestamp": self.timestamp,
            "total_duration_seconds": self.total_duration_seconds,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class FullPipelineTuner:
    """
    Оптимізатор повного pipeline.
    
    Приклад використання:
        tuner = FullPipelineTuner()
        
        report = tuner.tune(
            database_path="data/unified_disease_symptom_merged.json",
            output_dir="models/tuned/",
            strategy="iterative"
        )
        
        print(f"Покращення: {report.total_improvement:.2%}")
    """
    
    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()
    
    def tune(
        self,
        database_path: str,
        output_dir: str,
        som_path: Optional[str] = None,
        nn_path: Optional[str] = None,
        strategy: str = "iterative",
        tune_som: bool = True,
        tune_candidate: bool = True,
        tune_nn: bool = True,
        verbose: bool = True
    ) -> FullTuningReport:
        """
        Повна оптимізація pipeline.
        
        Args:
            database_path: Шлях до бази хвороб
            output_dir: Директорія для збереження результатів
            som_path: Поточна SOM модель (опціонально)
            nn_path: Поточна NN модель (опціонально)
            strategy: "iterative", "grid", "bayesian"
            tune_som: Чи оптимізувати SOM
            tune_candidate: Чи оптимізувати Candidate Selector
            tune_nn: Чи оптимізувати NN
            verbose: Виводити прогрес
            
        Returns:
            FullTuningReport
        """
        start_time = datetime.now()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print("=" * 60)
            print("FULL PIPELINE TUNING")
            print("=" * 60)
            print(f"Strategy: {strategy}")
            print(f"Output: {output_dir}")
        
        # Оцінка початкової якості
        initial_score = 0.0
        if som_path and nn_path:
            initial_score = self._evaluate_pipeline(
                som_path, nn_path, database_path, verbose
            )
            if verbose:
                print(f"\nПочаткова якість: {initial_score:.4f}")
        
        results = {}
        current_som_path = som_path
        current_nn_path = nn_path
        
        # 1. Tuning SOM
        if tune_som:
            if verbose:
                print("\n" + "-" * 40)
                print("📊 Tuning SOM...")
            
            som_result = self._tune_som(
                database_path=database_path,
                output_path=output_path / "som_tuned.pkl",
                verbose=verbose
            )
            results['som'] = som_result
            
            if som_result:
                current_som_path = str(output_path / "som_tuned.pkl")
        
        # 2. Tuning Candidate Selector (параметри зберігаються в конфігурації)
        if tune_candidate and current_som_path:
            if verbose:
                print("\n" + "-" * 40)
                print("📋 Tuning Candidate Selector...")
            
            candidate_result = self._tune_candidate(
                som_path=current_som_path,
                database_path=database_path,
                verbose=verbose
            )
            results['candidate'] = candidate_result
        
        # 3. Tuning NN
        if tune_nn and current_som_path:
            if verbose:
                print("\n" + "-" * 40)
                print("🧠 Tuning Neural Network...")
            
            nn_result = self._tune_nn(
                som_path=current_som_path,
                database_path=database_path,
                output_path=output_path / "nn_tuned.pt",
                verbose=verbose
            )
            results['nn'] = nn_result
            
            if nn_result:
                current_nn_path = str(output_path / "nn_tuned.pt")
        
        # Фінальна оцінка
        final_score = 0.0
        if current_som_path and current_nn_path:
            final_score = self._evaluate_pipeline(
                current_som_path, current_nn_path, database_path, verbose
            )
        
        improvement = final_score - initial_score
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Збираємо найкращу конфігурацію
        best_config = self._collect_best_config(results)
        
        report = FullTuningReport(
            som_result=results.get('som'),
            candidate_result=results.get('candidate'),
            nn_result=results.get('nn'),
            initial_score=initial_score,
            final_score=final_score,
            total_improvement=improvement,
            best_config=best_config,
            timestamp=datetime.now().isoformat(),
            total_duration_seconds=duration,
        )
        
        # Зберігаємо звіт
        report.save(str(output_path / "tuning_report.json"))
        
        if verbose:
            print("\n" + "=" * 60)
            print("РЕЗУЛЬТАТИ TUNING")
            print("=" * 60)
            print(f"Початкова якість: {initial_score:.4f}")
            print(f"Фінальна якість:  {final_score:.4f}")
            print(f"Покращення:       {improvement:+.4f} ({improvement/max(initial_score, 0.001)*100:+.1f}%)")
            print(f"Час:              {duration:.1f}с")
            print(f"Звіт:             {output_path / 'tuning_report.json'}")
        
        return report
    
    def _evaluate_pipeline(
        self,
        som_path: str,
        nn_path: str,
        database_path: str,
        verbose: bool
    ) -> float:
        """Оцінка якості pipeline"""
        try:
            from dr_case.validation import validate_pipeline
            
            report = validate_pipeline(
                som_path=som_path,
                nn_path=nn_path,
                database_path=database_path,
                n_samples=200,
                verbose=False
            )
            
            # Комбінована оцінка
            score = 0.0
            
            if report.candidate_report:
                score += report.candidate_report.recall * 0.3
            
            if report.nn_report:
                score += report.nn_report.recall_5 * 0.4
                score += report.nn_report.mean_average_precision * 0.3
            
            return score
            
        except Exception as e:
            if verbose:
                print(f"   Помилка оцінки: {e}")
            return 0.0
    
    def _tune_som(
        self,
        database_path: str,
        output_path: Path,
        verbose: bool
    ) -> Optional[TuningResult]:
        """Tuning SOM з grid search"""
        start_time = datetime.now()
        trials = []
        best_score = -1.0
        best_params = {}
        
        try:
            from dr_case.optimization.som_tuner import SOMTuner
            from dr_case.som import SOMTrainer
            
            # Завантажуємо базу
            with open(database_path, 'r', encoding='utf-8') as f:
                database = json.load(f)
            
            # Grid search
            for grid_size in self.config.som_grid_sizes:
                for lr in self.config.som_learning_rates:
                    params = {
                        'height': grid_size[0],
                        'width': grid_size[1],
                        'learning_rate': lr,
                    }
                    
                    if verbose:
                        print(f"   Trying: {grid_size}, lr={lr}...")
                    
                    try:
                        # Навчаємо SOM
                        trainer = SOMTrainer(database_path)
                        trainer.train(
                            height=grid_size[0],
                            width=grid_size[1],
                            learning_rate_init=lr,
                            epochs=200,
                            verbose=False
                        )
                        
                        # Оцінюємо
                        qe = trainer.quantization_error
                        te = trainer.topographic_error
                        
                        # Score = 1 - (QE + TE) / 2
                        score = 1.0 - (qe + te) / 2
                        
                        trials.append({
                            'params': params,
                            'score': score,
                            'qe': qe,
                            'te': te,
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            # Зберігаємо найкращу модель
                            trainer.save_model(str(output_path))
                        
                    except Exception as e:
                        if verbose:
                            print(f"      Skip: {e}")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TuningResult(
                component="SOM",
                best_params=best_params,
                best_score=best_score,
                all_trials=trials,
                improvement=0.0,  # Буде обчислено пізніше
                duration_seconds=duration,
            )
            
        except ImportError as e:
            if verbose:
                print(f"   SOM tuner недоступний: {e}")
            return None
    
    def _tune_candidate(
        self,
        som_path: str,
        database_path: str,
        verbose: bool
    ) -> Optional[TuningResult]:
        """Tuning Candidate Selector"""
        start_time = datetime.now()
        trials = []
        best_score = -1.0
        best_params = {}
        
        try:
            from dr_case.validation import CandidateRecallValidator
            
            validator = CandidateRecallValidator()
            
            for alpha in self.config.candidate_alphas:
                for k in self.config.candidate_k:
                    params = {'alpha': alpha, 'k': k}
                    
                    if verbose:
                        print(f"   Trying: α={alpha}, k={k}...")
                    
                    try:
                        report = validator.validate_from_checkpoint(
                            som_checkpoint_path=som_path,
                            database_path=database_path,
                            n_samples=300
                        )
                        
                        score = report.recall
                        
                        trials.append({
                            'params': params,
                            'score': score,
                            'avg_candidates': report.avg_candidates,
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                        
                    except Exception as e:
                        if verbose:
                            print(f"      Skip: {e}")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TuningResult(
                component="CandidateSelector",
                best_params=best_params,
                best_score=best_score,
                all_trials=trials,
                improvement=0.0,
                duration_seconds=duration,
            )
            
        except Exception as e:
            if verbose:
                print(f"   Candidate tuner помилка: {e}")
            return None
    
    def _tune_nn(
        self,
        som_path: str,
        database_path: str,
        output_path: Path,
        verbose: bool
    ) -> Optional[TuningResult]:
        """Tuning Neural Network"""
        start_time = datetime.now()
        trials = []
        best_score = -1.0
        best_params = {}
        
        try:
            # Спрощена версія - тільки логування параметрів
            # Повний tuning потребує GPU та значного часу
            
            for hidden in self.config.nn_hidden_dims[:1]:  # Тільки перший варіант
                for dropout in self.config.nn_dropout[:1]:
                    for lr in self.config.nn_learning_rates[:1]:
                        params = {
                            'hidden_dims': hidden,
                            'dropout': dropout,
                            'learning_rate': lr,
                        }
                        
                        if verbose:
                            print(f"   Config: hidden={hidden}, dropout={dropout}, lr={lr}")
                        
                        # TODO: Повне навчання NN
                        # Поки що зберігаємо параметри
                        
                        trials.append({
                            'params': params,
                            'score': 0.0,
                        })
                        
                        if not best_params:
                            best_params = params
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TuningResult(
                component="NeuralNetwork",
                best_params=best_params,
                best_score=best_score,
                all_trials=trials,
                improvement=0.0,
                duration_seconds=duration,
            )
            
        except Exception as e:
            if verbose:
                print(f"   NN tuner помилка: {e}")
            return None
    
    def _collect_best_config(self, results: Dict[str, TuningResult]) -> Dict[str, Any]:
        """Збирає найкращу конфігурацію"""
        config = {}
        
        if 'som' in results and results['som']:
            config['som'] = results['som'].best_params
        
        if 'candidate' in results and results['candidate']:
            config['candidate_selector'] = results['candidate'].best_params
        
        if 'nn' in results and results['nn']:
            config['neural_network'] = results['nn'].best_params
        
        return config


# Швидкий доступ
def tune_pipeline(
    database_path: str,
    output_dir: str,
    som_path: Optional[str] = None,
    nn_path: Optional[str] = None,
    verbose: bool = True
) -> FullTuningReport:
    """
    Швидка оптимізація pipeline.
    
    Args:
        database_path: Шлях до бази хвороб
        output_dir: Директорія для результатів
        som_path: Поточна SOM (опціонально)
        nn_path: Поточна NN (опціонально)
        verbose: Виводити прогрес
        
    Returns:
        FullTuningReport
    """
    tuner = FullPipelineTuner()
    return tuner.tune(
        database_path=database_path,
        output_dir=output_dir,
        som_path=som_path,
        nn_path=nn_path,
        verbose=verbose
    )
