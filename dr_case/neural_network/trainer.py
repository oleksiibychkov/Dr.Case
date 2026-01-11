"""
Dr.Case — Навчання нейронної мережі

Trainer для навчання DiagnosisNN на базі даних діагнозів.
Підтримує:
- Data augmentation (додавання/видалення симптомів)
- Early stopping
- Learning rate scheduling
- Збереження/завантаження checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .model import DiagnosisNN, FocalLoss, create_model


@dataclass
class TrainingConfig:
    """Конфігурація навчання"""
    # Архітектура
    hidden_dims: List[int] = None
    dropout: float = 0.3
    
    # Навчання
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Data augmentation
    augment: bool = True
    augment_prob: float = 0.3
    augment_drop_prob: float = 0.2
    augment_add_prob: float = 0.1
    
    # Loss
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class DiseaseDataset(Dataset):
    """
    Dataset для навчання.
    
    Кожен приклад: (symptom_vector, disease_index)
    """
    
    def __init__(
        self,
        disease_vectors: np.ndarray,
        disease_indices: np.ndarray,
        augment: bool = False,
        augment_prob: float = 0.3,
        drop_prob: float = 0.2,
        add_prob: float = 0.1,
        all_symptoms_count: int = 461
    ):
        """
        Args:
            disease_vectors: Матриця векторів діагнозів (N, D)
            disease_indices: Індекси діагнозів (N,)
            augment: Чи застосовувати аугментацію
            augment_prob: Ймовірність аугментації прикладу
            drop_prob: Ймовірність видалення симптому
            add_prob: Ймовірність додавання випадкового симптому
        """
        self.disease_vectors = torch.FloatTensor(disease_vectors)
        self.disease_indices = torch.LongTensor(disease_indices)
        self.augment = augment
        self.augment_prob = augment_prob
        self.drop_prob = drop_prob
        self.add_prob = add_prob
        self.n_symptoms = all_symptoms_count
    
    def __len__(self) -> int:
        return len(self.disease_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vector = self.disease_vectors[idx].clone()
        target = self.disease_indices[idx]
        
        # Аугментація
        if self.augment and np.random.random() < self.augment_prob:
            vector = self._augment(vector)
        
        return vector, target
    
    def _augment(self, vector: torch.Tensor) -> torch.Tensor:
        """Застосувати аугментацію до вектора"""
        # Знаходимо активні симптоми
        active = (vector > 0).nonzero(as_tuple=True)[0]
        inactive = (vector == 0).nonzero(as_tuple=True)[0]
        
        # Видаляємо випадкові симптоми
        if len(active) > 1:  # Залишаємо хоча б 1
            for idx in active:
                if np.random.random() < self.drop_prob:
                    vector[idx] = 0
        
        # Додаємо випадкові симптоми
        if len(inactive) > 0:
            for idx in inactive:
                if np.random.random() < self.add_prob:
                    vector[idx] = 1
        
        return vector


class NNTrainer:
    """
    Trainer для DiagnosisNN.
    
    Приклад використання:
        from dr_case.encoding import DiseaseEncoder
        
        # Підготовка даних
        encoder = DiseaseEncoder.from_database("data/unified_disease_symptom_data_full.json")
        disease_matrix = encoder.encode_all()
        
        # Навчання
        trainer = NNTrainer(n_symptoms=461, n_diseases=842)
        history = trainer.train(disease_matrix, encoder.disease_names)
        
        # Збереження
        trainer.save("models/nn_model.pt")
    """
    
    def __init__(
        self,
        n_symptoms: int = 461,
        n_diseases: int = 842,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            n_symptoms: Кількість симптомів
            n_diseases: Кількість діагнозів
            config: Конфігурація навчання
            device: 'cuda' або 'cpu'
        """
        self.n_symptoms = n_symptoms
        self.n_diseases = n_diseases
        self.config = config or TrainingConfig()
        
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Модель
        self.model = DiagnosisNN(
            n_symptoms=n_symptoms,
            n_diseases=n_diseases,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Disease name mapping
        self.disease_names: List[str] = []
        self.disease_to_idx: Dict[str, int] = {}
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        print(f"NNTrainer initialized on {self.device}")
        print(f"Model: {self.model.count_parameters():,} parameters")
    
    def _setup_training(self):
        """Налаштувати optimizer, scheduler, criterion"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(gamma=self.config.focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )
    
    def _create_dataloaders(
        self,
        disease_matrix: np.ndarray,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Створити train/val dataloaders"""
        n_samples = len(disease_matrix)
        indices = np.arange(n_samples)
        
        # Для валідації використовуємо ТІ Ж діагнози, але без аугментації
        # Це перевіряє чи модель правильно вивчила паттерни
        np.random.shuffle(indices)
        
        # Train dataset з аугментацією (всі дані)
        train_dataset = DiseaseDataset(
            disease_vectors=disease_matrix,
            disease_indices=np.arange(n_samples),
            augment=self.config.augment,
            augment_prob=self.config.augment_prob,
            drop_prob=self.config.augment_drop_prob,
            add_prob=self.config.augment_add_prob,
            all_symptoms_count=self.n_symptoms
        )
        
        # Val dataset - оригінальні дані без аугментації
        # Перевіряємо чи модель правильно класифікує "чисті" вектори
        val_dataset = DiseaseDataset(
            disease_vectors=disease_matrix,
            disease_indices=np.arange(n_samples),
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader 
    
    def train(
        self,
        disease_matrix: np.ndarray,
        disease_names: List[str],
        val_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Навчити модель.
        
        Args:
            disease_matrix: Матриця векторів діагнозів (N, D)
            disease_names: Список назв діагнозів
            val_split: Частка валідаційних даних
            verbose: Виводити прогрес
            
        Returns:
            Історія навчання {train_loss, val_loss, train_acc, val_acc}
        """
        # Зберігаємо mapping
        self.disease_names = disease_names
        self.disease_to_idx = {name: idx for idx, name in enumerate(disease_names)}
        
        # Setup
        self._setup_training()
        train_loader, val_loader = self._create_dataloaders(disease_matrix, val_split)
        
        print(f"Training: {len(train_loader.dataset)} samples (augmented)")
        print(f"Validation: {len(val_loader.dataset)} samples (original)")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            
            # History
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Scheduler
            if self.config.use_scheduler:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_loss - self.config.min_delta:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_best_model()
            else:
                self.epochs_without_improvement += 1
            
            # Print progress
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                      f"LR: {lr:.2e}")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Завантажуємо найкращу модель
        self._load_best_model()
        
        return history
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Одна епоха навчання"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for vectors, targets in loader:
            vectors = vectors.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(vectors)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * vectors.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += vectors.size(0)
        
        return total_loss / total, correct / total
    
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Валідація"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for vectors, targets in loader:
                vectors = vectors.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(vectors)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * vectors.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += vectors.size(0)
        
        return total_loss / total, correct / total
    
    def _save_best_model(self):
        """Зберегти найкращу модель у пам'яті"""
        self._best_state = {
            'model': self.model.state_dict().copy(),
            'optimizer': self.optimizer.state_dict().copy(),
        }
    
    def _load_best_model(self):
        """Завантажити найкращу модель"""
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state['model'])
    
    def save(self, path: str):
        """
        Зберегти модель та metadata.
        
        Args:
            path: Шлях до файлу .pt
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'n_symptoms': self.n_symptoms,
            'n_diseases': self.n_diseases,
            'config': asdict(self.config),
            'disease_names': self.disease_names,
            'disease_to_idx': self.disease_to_idx,
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "NNTrainer":
        """
        Завантажити модель.
        
        Args:
            path: Шлях до файлу .pt
            device: Device для моделі
            
        Returns:
            NNTrainer з завантаженою моделлю
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        config = TrainingConfig(**checkpoint['config'])
        
        trainer = cls(
            n_symptoms=checkpoint['n_symptoms'],
            n_diseases=checkpoint['n_diseases'],
            config=config,
            device=device
        )
        
        trainer.model.load_state_dict(checkpoint['model_state'])
        trainer.disease_names = checkpoint['disease_names']
        trainer.disease_to_idx = checkpoint['disease_to_idx']
        
        print(f"Model loaded from {path}")
        return trainer
    
    def predict(
        self,
        symptom_vector: np.ndarray,
        candidate_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Передбачити scores для вектора симптомів.
        
        Args:
            symptom_vector: Вектор симптомів (D,)
            candidate_indices: Індекси кандидатів (опціонально)
            
        Returns:
            (scores, indices) відсортовані за score
        """
        self.model.eval()
        
        with torch.no_grad():
            vector = torch.FloatTensor(symptom_vector).unsqueeze(0).to(self.device)
            
            if candidate_indices is not None:
                # Створюємо маску
                mask = torch.zeros(1, self.n_diseases).to(self.device)
                mask[0, candidate_indices] = 1
                scores = self.model(vector, mask)
            else:
                scores = self.model(vector)
            
            scores = scores.squeeze(0).cpu().numpy()
        
        # Сортуємо
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        
        return sorted_scores, sorted_indices
    
    def __repr__(self) -> str:
        return f"NNTrainer(model={self.model.count_parameters():,} params, device={self.device})"
