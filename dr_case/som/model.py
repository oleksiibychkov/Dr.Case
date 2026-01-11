"""
Dr.Case — Модель Self-Organizing Map

Обгортка над MiniSom з додатковою функціональністю:
- Збереження/завантаження моделі
- Відстеження діагнозів у юнітах
- Метрики якості (QE, TE)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from minisom import MiniSom

from dr_case.config import SOMConfig


class SOMModel:
    """
    Self-Organizing Map для кластеризації діагнозів.
    
    Кожен юніт карти може містити декілька діагнозів.
    При проєкції пацієнта знаходимо найближчі юніти та їх діагнози.
    
    Приклад використання:
        from dr_case.config import SOMConfig
        
        config = SOMConfig(grid_height=15, grid_width=15, input_dim=461)
        som = SOMModel(config)
        
        # Навчання
        som.train(disease_matrix, epochs=1000)
        
        # Проєкція пацієнта
        bmu = som.get_bmu(patient_vector)
        distances = som.get_distances(patient_vector)
    """
    
    def __init__(self, config: SOMConfig):
        """
        Args:
            config: Конфігурація SOM
        """
        self.config = config
        self._som: Optional[MiniSom] = None
        self._is_trained: bool = False
        
        # Мапінг юніт → діагнози
        self._unit_to_diseases: Dict[Tuple[int, int], List[str]] = {}
        
        # Мапінг діагноз → юніт
        self._disease_to_unit: Dict[str, Tuple[int, int]] = {}
        
        # Назви діагнозів (в порядку індексів)
        self._disease_names: List[str] = []
        
        # Метрики
        self._quantization_error: Optional[float] = None
        self._topographic_error: Optional[float] = None
    
    def _create_som(self) -> MiniSom:
        """Створити екземпляр MiniSom"""
        # Визначаємо функцію сусідства
        neighborhood = getattr(self.config, 'neighborhood_function', None)
        if neighborhood is not None and hasattr(neighborhood, 'value'):
            neighborhood = neighborhood.value
        else:
            neighborhood = "gaussian"
        
        # Отримуємо параметри з конфігу
        sigma = getattr(self.config, 'sigma_init', self.config.grid_height / 2)
        learning_rate = getattr(self.config, 'learning_rate_init', 0.5)
        
        som = MiniSom(
            x=self.config.grid_height,
            y=self.config.grid_width,
            input_len=self.config.input_dim,
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function=neighborhood,
            random_seed=42
        )
        
        return som
    
    def initialize(self, data: np.ndarray) -> None:
        """
        Ініціалізувати ваги SOM.
        
        Args:
            data: Матриця даних для ініціалізації shape (N, D)
        """
        self._som = self._create_som()
        
        init_method = getattr(self.config, 'initialization', None)
        if init_method is not None and hasattr(init_method, 'value'):
            init_method = init_method.value
        else:
            init_method = "pca"
        
        if init_method == "pca":
            self._som.pca_weights_init(data)
        else:
            self._som.random_weights_init(data)
    
    def train(
        self, 
        data: np.ndarray, 
        disease_names: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Навчити SOM.
        
        Args:
            data: Матриця даних shape (N_diseases, D_symptoms)
            disease_names: Список назв діагнозів (в порядку рядків data)
            epochs: Кількість епох (якщо None - з конфігу)
            verbose: Виводити прогрес
            
        Returns:
            Словник з метриками {qe, te, fill_ratio}
        """
        if epochs is None:
            epochs = getattr(self.config, 'epochs', 1000)
        
        # Зберігаємо назви
        if disease_names:
            self._disease_names = disease_names
        else:
            self._disease_names = [f"disease_{i}" for i in range(len(data))]
        
        # Ініціалізуємо якщо потрібно
        if self._som is None:
            self.initialize(data)
        
        # Навчаємо
        if verbose:
            print(f"Training SOM {self.config.grid_height}x{self.config.grid_width}...")
            print(f"  Data shape: {data.shape}")
            print(f"  Epochs: {epochs}")
        
        self._som.train(
            data, 
            epochs,
            verbose=verbose
        )
        
        self._is_trained = True
        
        # Будуємо мапінг юніт ↔ діагнози
        self._build_mappings(data)
        
        # Обчислюємо метрики
        metrics = self.compute_metrics(data)
        
        if verbose:
            print(f"  QE: {metrics['qe']:.4f}")
            print(f"  TE: {metrics['te']:.4f}")
            print(f"  Fill ratio: {metrics['fill_ratio']:.2%}")
        
        return metrics
    
    def _build_mappings(self, data: np.ndarray) -> None:
        """Побудувати мапінги юніт ↔ діагнози"""
        self._unit_to_diseases.clear()
        self._disease_to_unit.clear()
        
        for idx, vector in enumerate(data):
            bmu = self._som.winner(vector)
            disease_name = self._disease_names[idx]
            
            # Юніт → діагнози
            if bmu not in self._unit_to_diseases:
                self._unit_to_diseases[bmu] = []
            self._unit_to_diseases[bmu].append(disease_name)
            
            # Діагноз → юніт
            self._disease_to_unit[disease_name] = bmu
    
    def compute_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Обчислити метрики якості SOM.
        
        Args:
            data: Матриця даних
            
        Returns:
            Словник {qe, te, fill_ratio}
        """
        if not self._is_trained:
            raise RuntimeError("SOM is not trained yet")
        
        # Quantization Error - середня відстань до BMU
        self._quantization_error = self._som.quantization_error(data)
        
        # Topographic Error - частка випадків коли 2-й найближчий не є сусідом BMU
        self._topographic_error = self._som.topographic_error(data)
        
        # Fill ratio - частка заповнених юнітів
        total_units = self.config.grid_height * self.config.grid_width
        filled_units = len(self._unit_to_diseases)
        fill_ratio = filled_units / total_units
        
        return {
            "qe": self._quantization_error,
            "te": self._topographic_error,
            "fill_ratio": fill_ratio,
            "filled_units": filled_units,
            "total_units": total_units,
        }
    
    def get_bmu(self, vector: np.ndarray) -> Tuple[int, int]:
        """
        Знайти Best Matching Unit для вектора.
        
        Args:
            vector: Вектор симптомів shape (D,)
            
        Returns:
            Координати BMU (row, col)
        """
        if not self._is_trained:
            raise RuntimeError("SOM is not trained yet")
        
        return self._som.winner(vector)
    
    def get_bmu_distance(self, vector: np.ndarray) -> float:
        """
        Отримати відстань до BMU.
        
        Args:
            vector: Вектор симптомів
            
        Returns:
            Евклідова відстань до BMU
        """
        bmu = self.get_bmu(vector)
        weights = self._som.get_weights()
        bmu_weights = weights[bmu[0], bmu[1]]
        return np.linalg.norm(vector - bmu_weights)
    
    def get_all_distances(self, vector: np.ndarray) -> np.ndarray:
        """
        Отримати відстані до всіх юнітів.
        
        Args:
            vector: Вектор симптомів
            
        Returns:
            Матриця відстаней shape (H, W)
        """
        if not self._is_trained:
            raise RuntimeError("SOM is not trained yet")
        
        weights = self._som.get_weights()  # shape (H, W, D)
        
        # Обчислюємо відстані
        diff = weights - vector  # broadcasting
        distances = np.linalg.norm(diff, axis=2)  # shape (H, W)
        
        return distances
    
    def get_diseases_in_unit(self, unit: Tuple[int, int]) -> List[str]:
        """
        Отримати діагнози в юніті.
        
        Args:
            unit: Координати юніта (row, col)
            
        Returns:
            Список назв діагнозів
        """
        return self._unit_to_diseases.get(unit, [])
    
    def get_unit_for_disease(self, disease_name: str) -> Optional[Tuple[int, int]]:
        """
        Отримати юніт для діагнозу.
        
        Args:
            disease_name: Назва діагнозу
            
        Returns:
            Координати юніта або None
        """
        return self._disease_to_unit.get(disease_name)
    
    def get_neighbors(self, unit: Tuple[int, int], radius: int = 1) -> List[Tuple[int, int]]:
        """
        Отримати сусідні юніти.
        
        Args:
            unit: Центральний юніт
            radius: Радіус сусідства
            
        Returns:
            Список координат сусідів
        """
        neighbors = []
        row, col = unit
        
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = row + dr, col + dc
                
                if 0 <= nr < self.config.grid_height and 0 <= nc < self.config.grid_width:
                    neighbors.append((nr, nc))
        
        return neighbors
    
    def get_weights(self) -> np.ndarray:
        """
        Отримати ваги SOM.
        
        Returns:
            Матриця ваг shape (H, W, D)
        """
        if not self._is_trained:
            raise RuntimeError("SOM is not trained yet")
        
        return self._som.get_weights()
    
    @property
    def is_trained(self) -> bool:
        """Чи навчена модель"""
        return self._is_trained
    
    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Розмір карти (H, W)"""
        return (self.config.grid_height, self.config.grid_width)
    
    @property
    def filled_units(self) -> Set[Tuple[int, int]]:
        """Множина заповнених юнітів"""
        return set(self._unit_to_diseases.keys())
    
    @property
    def quantization_error(self) -> Optional[float]:
        """Quantization Error"""
        return self._quantization_error
    
    @property
    def topographic_error(self) -> Optional[float]:
        """Topographic Error"""
        return self._topographic_error
    
    def save(self, path: str) -> None:
        """
        Зберегти модель.
        
        Args:
            path: Шлях до файлу
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": self.config,
            "som": self._som,
            "unit_to_diseases": self._unit_to_diseases,
            "disease_to_unit": self._disease_to_unit,
            "disease_names": self._disease_names,
            "is_trained": self._is_trained,
            "qe": self._quantization_error,
            "te": self._topographic_error,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "SOMModel":
        """
        Завантажити модель.
        
        Args:
            path: Шлях до файлу
            
        Returns:
            SOMModel
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        model = cls(data["config"])
        model._som = data["som"]
        model._unit_to_diseases = data["unit_to_diseases"]
        model._disease_to_unit = data["disease_to_unit"]
        model._disease_names = data["disease_names"]
        model._is_trained = data["is_trained"]
        model._quantization_error = data["qe"]
        model._topographic_error = data["te"]
        
        return model
    
    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "not trained"
        return f"SOMModel({self.config.grid_height}x{self.config.grid_width}, {status})"
