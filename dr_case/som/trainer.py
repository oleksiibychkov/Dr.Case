"""
Dr.Case — Навчання SOM

Зручний інтерфейс для навчання SOM з бази даних діагнозів.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from dr_case.config import SOMConfig, get_default_config
from dr_case.encoding import DiseaseEncoder, DiseaseDatabaseLoader
from .model import SOMModel
from .projector import SOMProjector


class SOMTrainer:
    """
    Тренер SOM для бази даних діагнозів.
    
    Приклад використання:
        trainer = SOMTrainer()
        
        # Навчити з бази даних
        som, metrics = trainer.train_from_database("data/unified_disease_symptom_data_full.json")
        
        # Зберегти модель
        trainer.save_model("models/som_model.pkl")
        
        # Оцінити якість
        recall_metrics = trainer.evaluate_candidate_recall()
    """
    
    def __init__(self, config: Optional[SOMConfig] = None):
        """
        Args:
            config: Конфігурація SOM (якщо None - з default config)
        """
        if config is None:
            config = get_default_config().som
        
        self.config = config
        self.som_model: Optional[SOMModel] = None
        self.encoder: Optional[DiseaseEncoder] = None
        self.disease_matrix: Optional[np.ndarray] = None
        self.disease_names: Optional[list] = None
    
    def load_data(self, database_path: str) -> Tuple[np.ndarray, list]:
        """
        Завантажити та закодувати дані.
        
        Args:
            database_path: Шлях до JSON бази даних
            
        Returns:
            (матриця діагнозів, список назв)
        """
        print(f"Loading data from {database_path}...")
        
        self.encoder = DiseaseEncoder.from_database(database_path)
        
        # Кодуємо всі діагнози
        self.disease_matrix = self.encoder.encode_all(normalize=False)
        self.disease_names = self.encoder.disease_names
        
        print(f"  Diseases: {len(self.disease_names)}")
        print(f"  Symptoms: {self.encoder.vector_dim}")
        print(f"  Matrix shape: {self.disease_matrix.shape}")
        
        # Оновлюємо config якщо розміри відрізняються
        if self.config.input_dim != self.encoder.vector_dim:
            print(f"  Updating config input_dim: {self.config.input_dim} -> {self.encoder.vector_dim}")
            self.config.input_dim = self.encoder.vector_dim
        
        return self.disease_matrix, self.disease_names
    
    def train(
        self, 
        data: Optional[np.ndarray] = None,
        disease_names: Optional[list] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Навчити SOM.
        
        Args:
            data: Матриця даних (якщо None - використовує завантажені)
            disease_names: Назви діагнозів
            epochs: Кількість епох
            verbose: Виводити прогрес
            
        Returns:
            Метрики якості
        """
        if data is None:
            if self.disease_matrix is None:
                raise RuntimeError("No data loaded. Call load_data() first.")
            data = self.disease_matrix
            disease_names = self.disease_names
        
        # Створюємо модель
        self.som_model = SOMModel(self.config)
        
        # Навчаємо
        metrics = self.som_model.train(
            data=data,
            disease_names=disease_names,
            epochs=epochs,
            verbose=verbose
        )
        
        return metrics
    
    def train_from_database(
        self, 
        database_path: str,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[SOMModel, Dict[str, float]]:
        """
        Повний pipeline: завантажити дані та навчити SOM.
        
        Args:
            database_path: Шлях до JSON бази даних
            epochs: Кількість епох
            verbose: Виводити прогрес
            
        Returns:
            (навчена модель, метрики)
        """
        # Завантажуємо дані
        self.load_data(database_path)
        
        # Навчаємо
        metrics = self.train(epochs=epochs, verbose=verbose)
        
        return self.som_model, metrics
    
    def evaluate_candidate_recall(
        self,
        alpha: float = 0.9,
        k: int = 6,
        tau: float = 0.01
    ) -> Dict[str, float]:
        """
        Оцінити Candidate Recall.
        
        Для кожного діагнозу перевіряємо чи він потрапляє
        в кандидати при проєкції свого вектора.
        
        Args:
            alpha, k, tau: Параметри Candidate Selector
            
        Returns:
            Метрики recall
        """
        if self.som_model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        from dr_case.config import CandidateSelectorConfig
        
        selector_config = CandidateSelectorConfig(alpha=alpha, k=k, tau=tau)
        projector = SOMProjector(self.som_model, selector_config)
        
        metrics = projector.evaluate_recall(
            test_vectors=self.disease_matrix,
            true_diseases=self.disease_names
        )
        
        return metrics
    
    def get_projector(
        self,
        alpha: float = 0.9,
        k: int = 6,
        tau: float = 0.01
    ) -> SOMProjector:
        """
        Отримати проєктор для навченої моделі.
        
        Args:
            alpha, k, tau: Параметри Candidate Selector
            
        Returns:
            SOMProjector
        """
        if self.som_model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        from dr_case.config import CandidateSelectorConfig
        
        selector_config = CandidateSelectorConfig(alpha=alpha, k=k, tau=tau)
        return SOMProjector(self.som_model, selector_config)
    
    def save_model(self, path: str) -> None:
        """
        Зберегти модель.
        
        Args:
            path: Шлях до файлу
        """
        if self.som_model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.som_model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> SOMModel:
        """
        Завантажити модель.
        
        Args:
            path: Шлях до файлу
            
        Returns:
            SOMModel
        """
        self.som_model = SOMModel.load(path)
        self.config = self.som_model.config
        print(f"Model loaded from {path}")
        return self.som_model
    
    def visualize_map(self, save_path: Optional[str] = None):
        """
        Візуалізувати SOM карту.
        
        Args:
            save_path: Шлях для збереження зображення
        """
        if self.som_model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        import matplotlib.pyplot as plt
        
        # Кількість діагнозів в кожному юніті
        density = np.zeros(self.som_model.grid_shape)
        
        for unit, diseases in self.som_model._unit_to_diseases.items():
            density[unit[0], unit[1]] = len(diseases)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Density map
        ax1 = axes[0]
        im1 = ax1.imshow(density, cmap='YlOrRd', interpolation='nearest')
        ax1.set_title('Disease Density per Unit')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, label='Number of diseases')
        
        # 2. U-Matrix (unified distance matrix)
        weights = self.som_model.get_weights()
        H, W, D = weights.shape
        
        u_matrix = np.zeros((H, W))
        
        for i in range(H):
            for j in range(W):
                neighbors = self.som_model.get_neighbors((i, j), radius=1)
                if neighbors:
                    dists = []
                    for ni, nj in neighbors:
                        dist = np.linalg.norm(weights[i, j] - weights[ni, nj])
                        dists.append(dist)
                    u_matrix[i, j] = np.mean(dists)
        
        ax2 = axes[1]
        im2 = ax2.imshow(u_matrix, cmap='Blues', interpolation='nearest')
        ax2.set_title('U-Matrix (Average Distance to Neighbors)')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2, label='Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def __repr__(self) -> str:
        status = "trained" if self.som_model and self.som_model.is_trained else "not trained"
        return f"SOMTrainer({self.config.grid_height}x{self.config.grid_width}, {status})"
