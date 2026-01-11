"""
Dr.Case — Оптимізація параметрів Neural Network

Використовує Optuna для автоматичного підбору оптимальних параметрів:
- Архітектура (hidden_dims, dropout, use_batch_norm)
- Навчання (learning_rate, weight_decay, batch_size)
- Аугментація (augment_prob, drop_prob, add_prob)
- Loss function (focal loss vs cross-entropy)

Цільова функція: максимізувати Recall@K на валідаційній вибірці.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import numpy as np

try:
    import optuna
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dr_case.encoding import DiseaseEncoder
from dr_case.neural_network.model import DiagnosisNN, FocalLoss, create_model
from dr_case.neural_network.trainer import TrainingConfig, DiseaseDataset


@dataclass
class NNTuningResult:
    """Результат оптимізації Neural Network"""
    # Найкращі параметри
    best_params: Dict[str, Any]
    best_value: float
    
    # Метрики найкращої моделі
    best_metrics: Dict[str, float]
    
    # Історія всіх trials
    n_trials: int
    study_name: str
    
    # Конфігурація
    training_config: Optional[TrainingConfig] = None
    
    def to_dict(self) -> Dict:
        """Конвертувати в словник"""
        result = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_metrics": self.best_metrics,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
        }
        if self.training_config:
            result["training_config"] = asdict(self.training_config)
        return result
    
    def save(self, path: str) -> None:
        """Зберегти результат в JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "NNTuningResult":
        """Завантажити результат з JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Відновлюємо TrainingConfig якщо є
        if "training_config" in data:
            data["training_config"] = TrainingConfig(**data["training_config"])
        
        return cls(**data)


class NNTuner:
    """
    Оптимізатор параметрів Neural Network через Optuna.
    
    Приклад використання:
        tuner = NNTuner(database_path="data/unified_disease_symptom_data_full.json")
        
        # Запустити оптимізацію
        result = tuner.optimize(n_trials=50)
        
        print(f"Best Recall@5: {result.best_metrics['recall_at_5']:.2%}")
        print(f"Best params: {result.best_params}")
        
        # Отримати оптимальну модель
        model = tuner.get_best_model()
        
        # Навчити фінальну модель з найкращими параметрами
        final_model, metrics = tuner.train_with_best_params()
    """
    
    def __init__(
        self,
        database_path: str,
        target_recall_at_5: float = 0.95,
        target_recall_at_1: float = 0.70,
        validation_split: float = 0.2,
        max_epochs_per_trial: int = 50,
        random_seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Args:
            database_path: Шлях до JSON бази даних
            target_recall_at_5: Цільовий Recall@5 (за замовчуванням 95%)
            target_recall_at_1: Цільовий Recall@1 (за замовчуванням 70%)
            validation_split: Частка даних для валідації
            max_epochs_per_trial: Максимум епох на trial (для швидкості)
            random_seed: Seed для відтворюваності
            device: 'cuda' або 'cpu'
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        self.database_path = database_path
        self.target_recall_at_5 = target_recall_at_5
        self.target_recall_at_1 = target_recall_at_1
        self.validation_split = validation_split
        self.max_epochs_per_trial = max_epochs_per_trial
        self.random_seed = random_seed
        
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Завантажуємо дані
        self._load_data()
        
        # Зберігаємо найкращу модель
        self._best_model: Optional[DiagnosisNN] = None
        self._best_params: Optional[Dict] = None
        self._best_score: float = -float('inf')
    
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
        
        # Підготовка даних
        self.train_matrix = self.disease_matrix[self.train_indices]
        self.val_matrix = self.disease_matrix[self.val_indices]
        
        print(f"  Train: {len(self.train_indices)}, Validation: {len(self.val_indices)}")
        print(f"  Device: {self.device}")
    
    def _create_model(self, trial: Trial) -> DiagnosisNN:
        """Створити модель з параметрами trial"""
        # Архітектура
        n_layers = trial.suggest_int("n_layers", 2, 4)
        
        hidden_dims = []
        prev_dim = 512
        for i in range(n_layers):
            dim = trial.suggest_int(f"hidden_dim_{i}", 64, prev_dim, step=32)
            hidden_dims.append(dim)
            prev_dim = dim
        
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        
        model = DiagnosisNN(
            n_symptoms=self.n_symptoms,
            n_diseases=self.n_diseases,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        return model.to(self.device)
    
    def _create_dataloaders(
        self,
        trial: Trial,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Створити dataloaders з параметрами аугментації"""
        # Параметри аугментації
        augment_prob = trial.suggest_float("augment_prob", 0.1, 0.5)
        drop_prob = trial.suggest_float("drop_prob", 0.1, 0.4)
        add_prob = trial.suggest_float("add_prob", 0.05, 0.2)
        
        # Train dataset
        train_dataset = DiseaseDataset(
            disease_vectors=self.train_matrix,
            disease_indices=np.arange(len(self.train_indices)),
            augment=True,
            augment_prob=augment_prob,
            drop_prob=drop_prob,
            add_prob=add_prob,
            all_symptoms_count=self.n_symptoms
        )
        
        # Validation dataset (без аугментації)
        val_dataset = DiseaseDataset(
            disease_vectors=self.val_matrix,
            disease_indices=np.arange(len(self.val_indices)),
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def _compute_recall_at_k(
        self,
        model: DiagnosisNN,
        loader: DataLoader,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Обчислити Recall@K на датасеті"""
        model.eval()
        
        recalls = {k: 0 for k in k_values}
        total = 0
        
        with torch.no_grad():
            for vectors, targets in loader:
                vectors = vectors.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(vectors)
                
                for k in k_values:
                    _, top_k_preds = outputs.topk(k, dim=1)
                    
                    # Перевіряємо чи правильний клас в топ-K
                    for i, target in enumerate(targets):
                        if target.item() in top_k_preds[i].tolist():
                            recalls[k] += 1
                
                total += len(targets)
        
        return {f"recall_at_{k}": recalls[k] / total for k in k_values}
    
    def _train_epoch(
        self,
        model: DiagnosisNN,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Одна епоха навчання"""
        model.train()
        total_loss = 0
        total = 0
        
        for vectors, targets in loader:
            vectors = vectors.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(vectors)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * vectors.size(0)
            total += vectors.size(0)
        
        return total_loss / total
    
    def _create_objective(self) -> callable:
        """Створити objective функцію для Optuna"""
        
        def objective(trial: Trial) -> float:
            # Параметри навчання
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            
            # Focal Loss параметри
            use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
            focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0) if use_focal_loss else 2.0
            
            # Створюємо модель
            model = self._create_model(trial)
            
            # Dataloaders
            train_loader, val_loader = self._create_dataloaders(trial, batch_size)
            
            # Optimizer та Loss
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            if use_focal_loss:
                criterion = FocalLoss(gamma=focal_gamma)
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs_per_trial,
                eta_min=1e-6
            )
            
            # Training loop з early stopping
            best_val_recall = 0
            patience_counter = 0
            patience = 7
            best_state = model.state_dict().copy()  # Ініціалізація
            
            for epoch in range(self.max_epochs_per_trial):
                # Train
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
                
                # Validate
                val_metrics = self._compute_recall_at_k(model, val_loader)
                val_recall_5 = val_metrics["recall_at_5"]
                
                scheduler.step()
                
                # Early stopping
                if val_recall_5 > best_val_recall:
                    best_val_recall = val_recall_5
                    patience_counter = 0
                    
                    # Зберігаємо state якщо це найкращий результат
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Pruning - зупиняємо trial якщо результат поганий
                trial.report(val_recall_5, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                if patience_counter >= patience:
                    break
            
            # Завантажуємо найкращий state
            model.load_state_dict(best_state)
            
            # Фінальна оцінка
            final_metrics = self._compute_recall_at_k(model, val_loader)
            recall_at_1 = final_metrics["recall_at_1"]
            recall_at_5 = final_metrics["recall_at_5"]
            recall_at_10 = final_metrics["recall_at_10"]
            
            # Зберігаємо метрики
            trial.set_user_attr("recall_at_1", recall_at_1)
            trial.set_user_attr("recall_at_5", recall_at_5)
            trial.set_user_attr("recall_at_10", recall_at_10)
            trial.set_user_attr("n_params", model.count_parameters())
            trial.set_user_attr("hidden_dims", model.hidden_dims)
            
            # Score: комбінація Recall@5 (основна) та Recall@1 (бонус)
            score = recall_at_5 * 0.7 + recall_at_1 * 0.3
            
            # Зберігаємо найкращу модель
            if score > self._best_score:
                self._best_score = score
                self._best_model = model
                self._best_params = trial.params.copy()
                self._best_params["hidden_dims"] = model.hidden_dims
            
            return score
        
        return objective
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: str = "nn_optimization",
        show_progress: bool = True
    ) -> NNTuningResult:
        """
        Запустити оптимізацію.
        
        Args:
            n_trials: Кількість trials
            timeout: Ліміт часу в секундах
            study_name: Назва study для Optuna
            show_progress: Показувати прогрес
            
        Returns:
            NNTuningResult з найкращими параметрами
        """
        print(f"\n{'='*60}")
        print(f"Neural Network Hyperparameter Optimization")
        print(f"{'='*60}")
        print(f"  Trials: {n_trials}")
        print(f"  Max epochs per trial: {self.max_epochs_per_trial}")
        print(f"  Target Recall@5: {self.target_recall_at_5:.1%}")
        print(f"  Target Recall@1: {self.target_recall_at_1:.1%}")
        print(f"{'='*60}\n")
        
        # Створюємо study з pruning
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Callback для прогресу
        def progress_callback(study, trial):
            if show_progress and trial.number % 5 == 0:
                r1 = trial.user_attrs.get('recall_at_1', 0)
                r5 = trial.user_attrs.get('recall_at_5', 0)
                print(f"  Trial {trial.number}: score={trial.value:.4f}, "
                      f"R@1={r1:.2%}, R@5={r5:.2%}")
        
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
            "recall_at_1": best_trial.user_attrs.get("recall_at_1", 0),
            "recall_at_5": best_trial.user_attrs.get("recall_at_5", 0),
            "recall_at_10": best_trial.user_attrs.get("recall_at_10", 0),
            "n_params": best_trial.user_attrs.get("n_params", 0),
            "hidden_dims": best_trial.user_attrs.get("hidden_dims", []),
        }
        
        # Створюємо TrainingConfig
        bp = best_trial.params
        training_config = TrainingConfig(
            hidden_dims=best_metrics["hidden_dims"],
            dropout=bp.get("dropout", 0.3),
            learning_rate=bp.get("learning_rate", 1e-3),
            weight_decay=bp.get("weight_decay", 1e-4),
            batch_size=bp.get("batch_size", 64),
            epochs=100,  # Для фінального навчання
            augment=True,
            augment_prob=bp.get("augment_prob", 0.3),
            augment_drop_prob=bp.get("drop_prob", 0.2),
            augment_add_prob=bp.get("add_prob", 0.1),
            use_focal_loss=bp.get("use_focal_loss", False),
            focal_gamma=bp.get("focal_gamma", 2.0),
        )
        
        result = NNTuningResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_metrics=best_metrics,
            n_trials=len(study.trials),
            study_name=study_name,
            training_config=training_config
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Optimization complete!")
        print(f"{'='*60}")
        print(f"  Best score: {result.best_value:.4f}")
        print(f"  Recall@1: {best_metrics['recall_at_1']:.2%}")
        print(f"  Recall@5: {best_metrics['recall_at_5']:.2%}")
        print(f"  Recall@10: {best_metrics['recall_at_10']:.2%}")
        print(f"  Architecture: {best_metrics['hidden_dims']}")
        print(f"  Parameters: {best_metrics['n_params']:,}")
        print(f"{'='*60}")
        
        return result
    
    def get_best_model(self) -> DiagnosisNN:
        """
        Отримати найкращу модель після оптимізації.
        
        Returns:
            DiagnosisNN
        """
        if self._best_model is None:
            raise RuntimeError("No optimization run yet. Call optimize() first.")
        
        return self._best_model
    
    def train_with_best_params(
        self,
        params: Optional[Dict] = None,
        epochs: int = 100,
        verbose: bool = True
    ) -> Tuple[DiagnosisNN, Dict[str, float]]:
        """
        Навчити модель з найкращими параметрами на повних даних.
        
        Args:
            params: Параметри (якщо None - використовує best_params)
            epochs: Кількість епох для навчання
            verbose: Виводити прогрес
            
        Returns:
            (DiagnosisNN, metrics)
        """
        if params is None:
            if self._best_params is None:
                raise RuntimeError("No best params. Call optimize() first.")
            params = self._best_params
        
        print(f"\n{'='*60}")
        print(f"Training final model with best parameters")
        print(f"{'='*60}")
        
        # Створюємо модель
        hidden_dims = params.get("hidden_dims", [512, 256, 128])
        dropout = params.get("dropout", 0.3)
        use_batch_norm = params.get("use_batch_norm", True)
        
        model = DiagnosisNN(
            n_symptoms=self.n_symptoms,
            n_diseases=self.n_diseases,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        ).to(self.device)
        
        print(f"  Architecture: {hidden_dims}")
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Optimizer та Loss
        learning_rate = params.get("learning_rate", 1e-3)
        weight_decay = params.get("weight_decay", 1e-4)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        use_focal_loss = params.get("use_focal_loss", False)
        focal_gamma = params.get("focal_gamma", 2.0)
        
        if use_focal_loss:
            criterion = FocalLoss(gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        # Dataset на ВСІХ даних
        batch_size = params.get("batch_size", 64)
        augment_prob = params.get("augment_prob", 0.3)
        drop_prob = params.get("drop_prob", 0.2)
        add_prob = params.get("add_prob", 0.1)
        
        train_dataset = DiseaseDataset(
            disease_vectors=self.disease_matrix,
            disease_indices=np.arange(self.n_diseases),
            augment=True,
            augment_prob=augment_prob,
            drop_prob=drop_prob,
            add_prob=add_prob,
            all_symptoms_count=self.n_symptoms
        )
        
        val_dataset = DiseaseDataset(
            disease_vectors=self.disease_matrix,
            disease_indices=np.arange(self.n_diseases),
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training
        best_recall = 0
        best_state = None
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            metrics = self._compute_recall_at_k(model, val_loader)
            
            scheduler.step()
            
            if metrics["recall_at_5"] > best_recall:
                best_recall = metrics["recall_at_5"]
                best_state = model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                      f"R@1: {metrics['recall_at_1']:.2%} | "
                      f"R@5: {metrics['recall_at_5']:.2%} | "
                      f"LR: {lr:.2e}")
        
        # Завантажуємо найкращий state
        model.load_state_dict(best_state)
        
        # Фінальна оцінка
        final_metrics = self._compute_recall_at_k(model, val_loader)
        
        print(f"\n{'='*60}")
        print(f"✓ Training complete!")
        print(f"{'='*60}")
        print(f"  Final Recall@1: {final_metrics['recall_at_1']:.2%}")
        print(f"  Final Recall@5: {final_metrics['recall_at_5']:.2%}")
        print(f"  Final Recall@10: {final_metrics['recall_at_10']:.2%}")
        print(f"{'='*60}")
        
        return model, final_metrics
    
    def quick_tune(self, n_trials: int = 20) -> NNTuningResult:
        """
        Швидка оптимізація з меншою кількістю trials.
        
        Args:
            n_trials: Кількість trials
            
        Returns:
            NNTuningResult
        """
        return self.optimize(n_trials=n_trials, study_name="nn_quick_tune")
    
    def save_best_model(self, path: str) -> None:
        """
        Зберегти найкращу модель.
        
        Args:
            path: Шлях до файлу .pt
        """
        if self._best_model is None:
            raise RuntimeError("No best model. Call optimize() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state": self._best_model.state_dict(),
            "n_symptoms": self.n_symptoms,
            "n_diseases": self.n_diseases,
            "hidden_dims": self._best_model.hidden_dims,
            "dropout": self._best_model.dropout,
            "use_batch_norm": self._best_model.use_batch_norm,
            "best_params": self._best_params,
            "disease_names": self.disease_names,
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> DiagnosisNN:
        """
        Завантажити модель.
        
        Args:
            path: Шлях до файлу .pt
            
        Returns:
            DiagnosisNN
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        model = DiagnosisNN(
            n_symptoms=checkpoint["n_symptoms"],
            n_diseases=checkpoint["n_diseases"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            use_batch_norm=checkpoint["use_batch_norm"]
        ).to(self.device)
        
        model.load_state_dict(checkpoint["model_state"])
        print(f"Model loaded from {path}")
        
        return model
    
    def __repr__(self) -> str:
        return (
            f"NNTuner(diseases={self.n_diseases}, symptoms={self.n_symptoms}, "
            f"device={self.device})"
        )
