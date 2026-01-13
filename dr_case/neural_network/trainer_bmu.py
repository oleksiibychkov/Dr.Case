"""
Dr.Case — Навчання TwoBranchNN з BMU координатами

Архітектура:
  - symptoms: [n_symptoms] → branch → [128]
  - bmu_coords: [2] (row/H, col/W) → branch → [16]
  - combined → [128] → [n_diseases]

При навчанні:
  symptoms + BMU координати → правильний діагноз

При інференсі:
  Топ-k діагнозів, можна розділяти на "з кластера" і "не з кластера"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm

from dr_case.data_generation import FrequencySampler, SamplerConfig
from dr_case.som import SOMModel


@dataclass
class BMUTrainingConfig:
    """Конфігурація навчання TwoBranchNN з BMU"""
    
    # Архітектура
    symptom_hidden: List[int] = None
    bmu_hidden: List[int] = None
    combined_hidden: List[int] = None
    dropout: float = 0.3
    
    # Навчання
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Дані
    samples_per_disease: int = 100
    train_ratio: float = 0.8
    
    # Data augmentation
    noise_probability: float = 0.02
    dropout_probability: float = 0.15
    
    def __post_init__(self):
        if self.symptom_hidden is None:
            self.symptom_hidden = [256, 128]
        if self.bmu_hidden is None:
            self.bmu_hidden = [32, 16]
        if self.combined_hidden is None:
            self.combined_hidden = [128]


class TwoBranchNN_BMU(nn.Module):
    """TwoBranchNN з координатами BMU (2 числа)"""
    
    def __init__(
        self,
        n_symptoms: int,
        n_diseases: int,
        symptom_hidden: List[int] = None,
        bmu_hidden: List[int] = None,
        combined_hidden: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        symptom_hidden = symptom_hidden or [256, 128]
        bmu_hidden = bmu_hidden or [32, 16]
        combined_hidden = combined_hidden or [128]
        
        self.n_symptoms = n_symptoms
        self.n_diseases = n_diseases
        
        # Symptom branch
        layers = []
        in_dim = n_symptoms
        for out_dim in symptom_hidden:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.symptom_branch = nn.Sequential(*layers)
        
        # BMU coordinates branch (2 входи)
        layers = []
        in_dim = 2
        for out_dim in bmu_hidden:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ])
            in_dim = out_dim
        self.bmu_branch = nn.Sequential(*layers)
        
        # Combined
        combined_in = symptom_hidden[-1] + bmu_hidden[-1]
        layers = []
        in_dim = combined_in
        for out_dim in combined_hidden:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.combined = nn.Sequential(*layers)
        
        # Output
        self.output = nn.Linear(combined_hidden[-1], n_diseases)
        
        self.config = {
            'n_symptoms': n_symptoms,
            'n_diseases': n_diseases,
            'som_dim': 2,
            'symptom_hidden': symptom_hidden,
            'bmu_hidden': bmu_hidden,
            'combined_hidden': combined_hidden,
            'dropout': dropout,
        }
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, symptoms: torch.Tensor, bmu_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            symptoms: [batch, n_symptoms] бінарний вектор
            bmu_coords: [batch, 2] нормалізовані координати (row/H, col/W)
        """
        symptoms = nn.functional.normalize(symptoms, p=2, dim=-1)
        
        h_sym = self.symptom_branch(symptoms)
        h_bmu = self.bmu_branch(bmu_coords)
        
        h = torch.cat([h_sym, h_bmu], dim=-1)
        h = self.combined(h)
        
        return self.output(h)


class BMUDataGenerator:
    """Генератор даних для навчання з BMU координатами"""
    
    def __init__(
        self,
        database_path: str,
        som_path: str,
        config: BMUTrainingConfig
    ):
        self.config = config
        
        # Завантажуємо SOM
        self.som_model = SOMModel.load(som_path)
        self.minisom = self.som_model._som  # MiniSom object
        self.som_shape = self.minisom._weights.shape[:2]  # (H, W)
        
        # FrequencySampler
        sampler_config = SamplerConfig(
            samples_per_disease=config.samples_per_disease,
            min_symptoms=2,
            noise_probability=config.noise_probability,
            dropout_probability=config.dropout_probability,
            random_seed=42,
        )
        self.sampler = FrequencySampler.from_database(database_path, sampler_config)
        
        self.disease_names = self.sampler.disease_names
        self.symptom_names = self.sampler.symptom_names
        self.disease_to_idx = {name: i for i, name in enumerate(self.disease_names)}
        
        self.n_symptoms = len(self.symptom_names)
        self.n_diseases = len(self.disease_names)
    
    def _get_bmu_coords(self, symptom_vector: np.ndarray) -> np.ndarray:
        """Отримати нормалізовані координати BMU"""
        bmu = self.minisom.winner(symptom_vector)
        H, W = self.som_shape
        return np.array([bmu[0] / H, bmu[1] / W], dtype=np.float32)
    
    def generate(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерація даних.
        
        Returns:
            X_symptoms: [N, n_symptoms]
            X_bmu: [N, 2]
            y: [N]
        """
        samples = self.sampler.generate_samples(
            self.config.samples_per_disease,
            self.disease_names
        )
        
        total = len(samples)
        
        if verbose:
            print(f"Generating {total} samples with BMU coordinates...")
        
        X_symptoms = np.zeros((total, self.n_symptoms), dtype=np.float32)
        X_bmu = np.zeros((total, 2), dtype=np.float32)
        y = np.zeros(total, dtype=np.int64)
        
        for i, sample in enumerate(tqdm(samples, disable=not verbose)):
            X_symptoms[i] = sample.symptom_vector
            X_bmu[i] = self._get_bmu_coords(sample.symptom_vector)
            y[i] = self.disease_to_idx[sample.diagnosis]
        
        return X_symptoms, X_bmu, y


class BMUTrainer:
    """Trainer для TwoBranchNN з BMU координатами"""
    
    def __init__(
        self,
        database_path: str,
        som_path: str,
        config: Optional[BMUTrainingConfig] = None,
        device: str = None
    ):
        self.config = config or BMUTrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Генератор даних
        self.data_generator = BMUDataGenerator(database_path, som_path, self.config)
        
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top5': []}
    
    def _create_dataloaders(
        self,
        X_symptoms: np.ndarray,
        X_bmu: np.ndarray,
        y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Створення train/val DataLoaders"""
        
        n = len(y)
        idx = np.random.permutation(n)
        train_end = int(n * self.config.train_ratio)
        
        train_idx, val_idx = idx[:train_end], idx[train_end:]
        
        train_ds = TensorDataset(
            torch.FloatTensor(X_symptoms[train_idx]),
            torch.FloatTensor(X_bmu[train_idx]),
            torch.LongTensor(y[train_idx])
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_symptoms[val_idx]),
            torch.FloatTensor(X_bmu[val_idx]),
            torch.LongTensor(y[val_idx])
        )
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Навчання моделі.
        
        Returns:
            Метрики: val_acc, val_top5, etc.
        """
        # Генеруємо дані
        if verbose:
            print(f"Device: {self.device}")
            print(f"SOM shape: {self.data_generator.som_shape}")
            print(f"Diseases: {self.data_generator.n_diseases}")
            print(f"Symptoms: {self.data_generator.n_symptoms}")
        
        X_symptoms, X_bmu, y = self.data_generator.generate(verbose)
        
        train_loader, val_loader = self._create_dataloaders(X_symptoms, X_bmu, y)
        
        # Модель
        self.model = TwoBranchNN_BMU(
            n_symptoms=self.data_generator.n_symptoms,
            n_diseases=self.data_generator.n_diseases,
            symptom_hidden=self.config.symptom_hidden,
            bmu_hidden=self.config.bmu_hidden,
            combined_hidden=self.config.combined_hidden,
            dropout=self.config.dropout,
        ).to(self.device)
        
        if verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {total_params:,}")
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.config.epochs
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        best_top5_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_loss = 0
            
            for symptoms, bmu, targets in train_loader:
                symptoms = symptoms.to(self.device)
                bmu = bmu.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(symptoms, bmu)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            scheduler.step()
            
            # History
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_top5'].append(val_metrics['top5'])
            
            # Best model
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_top5_acc = val_metrics['top5']
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"Epoch {epoch+1}/{self.config.epochs}: "
                      f"Loss={train_loss/len(train_loader):.4f}, "
                      f"Val={val_metrics['acc']:.1f}%, "
                      f"Top5={val_metrics['top5']:.1f}%, "
                      f"Top10={val_metrics['top10']:.1f}%")
            
            # Early stopping
            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best
        self.model.load_state_dict(best_state)
        
        return {
            'val_acc': best_val_acc,
            'val_top5': best_top5_acc,
            'epochs_trained': epoch + 1,
        }
    
    def _validate(self, val_loader: DataLoader) -> Dict:
        """Валідація моделі"""
        self.model.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_top5_correct = 0
        val_top10_correct = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for symptoms, bmu, targets in val_loader:
                symptoms = symptoms.to(self.device)
                bmu = bmu.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(symptoms, bmu)
                val_loss += criterion(outputs, targets).item()
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                _, top5 = outputs.topk(5, dim=1)
                val_top5_correct += sum(
                    targets[i] in top5[i] for i in range(len(targets))
                )
                
                _, top10 = outputs.topk(10, dim=1)
                val_top10_correct += sum(
                    targets[i] in top10[i] for i in range(len(targets))
                )
        
        return {
            'loss': val_loss / len(val_loader),
            'acc': 100. * val_correct / val_total,
            'top5': 100. * val_top5_correct / val_total,
            'top10': 100. * val_top10_correct / val_total,
        }
    
    def save(self, path: str):
        """Зберегти модель"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Symptom names в lowercase!
        symptom_names_lower = [s.lower() for s in self.data_generator.symptom_names]
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'model_config': self.model.config,
            'disease_names': self.data_generator.disease_names,
            'symptom_names': symptom_names_lower,
            'som_shape': self.data_generator.som_shape,
            'training_config': asdict(self.config),
            'history': self.history,
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = None) -> TwoBranchNN_BMU:
        """Завантажити модель з checkpoint"""
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = TwoBranchNN_BMU(
            n_symptoms=config['n_symptoms'],
            n_diseases=config['n_diseases'],
            symptom_hidden=config['symptom_hidden'],
            bmu_hidden=config['bmu_hidden'],
            combined_hidden=config['combined_hidden'],
            dropout=config['dropout'],
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Attach metadata
        model.disease_names = checkpoint['disease_names']
        model.symptom_names = checkpoint['symptom_names']
        model.som_shape = checkpoint['som_shape']
        
        return model


def train_and_save(
    database_path: str = 'data/unified_disease_symptom_merged.json',
    som_path: str = 'models/som_merged.pkl',
    output_path: str = 'models/nn_two_branch_bmu.pt',
    **kwargs
) -> Dict:
    """
    Convenience function для навчання і збереження.
    
    Example:
        metrics = train_and_save(samples_per_disease=100, epochs=50)
    """
    config = BMUTrainingConfig(**kwargs)
    
    trainer = BMUTrainer(database_path, som_path, config)
    metrics = trainer.train(verbose=True)
    trainer.save(output_path)
    
    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("TwoBranchNN Training with BMU Coordinates")
    print("=" * 60)
    
    metrics = train_and_save(
        samples_per_disease=100,
        epochs=50,
    )
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"  Val Accuracy: {metrics['val_acc']:.1f}%")
    print(f"  Val Top-5: {metrics['val_top5']:.1f}%")
    print("=" * 60)
