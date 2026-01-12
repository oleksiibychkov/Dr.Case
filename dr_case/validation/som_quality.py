"""
Dr.Case — Метрики якості SOM

Метрики:
- QE (Quantization Error) — середня відстань до BMU
- TE (Topographic Error) — частка порушень топології
- Fill Rate — заповненість карти
- Cluster Distribution — розподіл діагнозів по юнітах
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum


class QualityLevel(Enum):
    """Рівень якості"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class SOMQualityThresholds:
    """Пороги якості SOM"""
    # Quantization Error
    qe_excellent: float = 0.2
    qe_good: float = 0.3
    qe_acceptable: float = 0.5
    
    # Topographic Error
    te_excellent: float = 0.05
    te_good: float = 0.1
    te_acceptable: float = 0.2
    
    # Fill Rate
    fill_min: float = 0.6
    fill_max: float = 0.95
    
    # Diagnoses per unit
    diagnoses_per_unit_min: int = 2
    diagnoses_per_unit_max: int = 10


@dataclass
class SOMQualityReport:
    """Звіт про якість SOM"""
    # Основні метрики
    quantization_error: float
    topographic_error: float
    fill_rate: float
    
    # Рівні якості
    qe_level: QualityLevel
    te_level: QualityLevel
    fill_level: QualityLevel
    overall_level: QualityLevel
    
    # Статистика кластерів
    total_units: int
    active_units: int
    empty_units: int
    
    # Розподіл діагнозів
    diagnoses_per_unit_mean: float
    diagnoses_per_unit_std: float
    diagnoses_per_unit_min: int
    diagnoses_per_unit_max: int
    
    # Діагнози без юнітів
    orphan_diagnoses: List[str] = field(default_factory=list)
    
    # Додаткова інформація
    grid_size: Tuple[int, int] = (0, 0)
    total_diagnoses: int = 0
    
    def is_acceptable(self) -> bool:
        """Чи прийнятна якість"""
        return self.overall_level in [
            QualityLevel.EXCELLENT,
            QualityLevel.GOOD,
            QualityLevel.ACCEPTABLE
        ]
    
    def to_dict(self) -> dict:
        """Серіалізація"""
        return {
            "quantization_error": self.quantization_error,
            "topographic_error": self.topographic_error,
            "fill_rate": self.fill_rate,
            "qe_level": self.qe_level.value,
            "te_level": self.te_level.value,
            "fill_level": self.fill_level.value,
            "overall_level": self.overall_level.value,
            "total_units": self.total_units,
            "active_units": self.active_units,
            "empty_units": self.empty_units,
            "diagnoses_per_unit": {
                "mean": self.diagnoses_per_unit_mean,
                "std": self.diagnoses_per_unit_std,
                "min": self.diagnoses_per_unit_min,
                "max": self.diagnoses_per_unit_max,
            },
            "orphan_diagnoses": self.orphan_diagnoses,
            "grid_size": self.grid_size,
            "total_diagnoses": self.total_diagnoses,
            "is_acceptable": self.is_acceptable(),
        }
    
    def __repr__(self) -> str:
        return (
            f"SOMQualityReport(\n"
            f"  QE={self.quantization_error:.4f} ({self.qe_level.value})\n"
            f"  TE={self.topographic_error:.4f} ({self.te_level.value})\n"
            f"  Fill={self.fill_rate:.2%} ({self.fill_level.value})\n"
            f"  Overall: {self.overall_level.value}\n"
            f"  Units: {self.active_units}/{self.total_units} active\n"
            f"  Diagnoses/unit: {self.diagnoses_per_unit_mean:.1f} ± {self.diagnoses_per_unit_std:.1f}\n"
            f")"
        )


class SOMQualityValidator:
    """
    Валідатор якості SOM.
    
    Приклад використання:
        validator = SOMQualityValidator()
        report = validator.validate(som_model, disease_vectors)
        
        if report.is_acceptable():
            print("SOM якість прийнятна")
        else:
            print(f"Проблеми: QE={report.qe_level}, TE={report.te_level}")
    """
    
    def __init__(self, thresholds: Optional[SOMQualityThresholds] = None):
        self.thresholds = thresholds or SOMQualityThresholds()
    
    def validate(
        self,
        som_model: Any,
        disease_vectors: Optional[Dict[str, np.ndarray]] = None,
        unit_to_diseases: Optional[Dict[int, List[str]]] = None
    ) -> SOMQualityReport:
        """
        Повна валідація SOM.
        
        Args:
            som_model: SOM модель (MiniSom або SOMModel)
            disease_vectors: {disease_name: vector} для обчислення QE/TE
            unit_to_diseases: {unit_idx: [diseases]} для статистики кластерів
            
        Returns:
            SOMQualityReport
        """
        # Визначаємо тип моделі
        from minisom import MiniSom
        
        if isinstance(som_model, MiniSom):
            return self._validate_minisom(som_model, disease_vectors, unit_to_diseases)
        elif hasattr(som_model, '_som'):
            # SOMModel wrapper
            return self._validate_som_model(som_model)
        else:
            raise ValueError(f"Невідомий тип SOM: {type(som_model)}")
    
    def _validate_minisom(
        self,
        som: Any,
        disease_vectors: Optional[Dict[str, np.ndarray]],
        unit_to_diseases: Optional[Dict[int, List[str]]]
    ) -> SOMQualityReport:
        """Валідація MiniSom"""
        from minisom import MiniSom
        
        # Розміри карти
        h, w = som._weights.shape[:2]
        total_units = h * w
        
        # QE та TE
        if disease_vectors:
            vectors = np.array(list(disease_vectors.values()))
            qe = som.quantization_error(vectors)
            te = som.topographic_error(vectors)
            total_diagnoses = len(disease_vectors)
        else:
            qe = 0.0
            te = 0.0
            total_diagnoses = 0
        
        # Статистика юнітів
        if unit_to_diseases:
            active_units = len([u for u, d in unit_to_diseases.items() if len(d) > 0])
            empty_units = total_units - active_units
            
            counts = [len(d) for d in unit_to_diseases.values() if len(d) > 0]
            if counts:
                mean_per_unit = np.mean(counts)
                std_per_unit = np.std(counts)
                min_per_unit = min(counts)
                max_per_unit = max(counts)
            else:
                mean_per_unit = std_per_unit = 0.0
                min_per_unit = max_per_unit = 0
            
            # Orphan diagnoses (не в жодному юніті)
            all_in_units = set()
            for diseases in unit_to_diseases.values():
                all_in_units.update(diseases)
            
            if disease_vectors:
                orphans = [d for d in disease_vectors.keys() if d not in all_in_units]
            else:
                orphans = []
        else:
            active_units = 0
            empty_units = total_units
            mean_per_unit = std_per_unit = 0.0
            min_per_unit = max_per_unit = 0
            orphans = []
        
        # Fill rate
        fill_rate = active_units / total_units if total_units > 0 else 0.0
        
        # Рівні якості
        qe_level = self._classify_qe(qe)
        te_level = self._classify_te(te)
        fill_level = self._classify_fill(fill_rate)
        overall_level = self._compute_overall(qe_level, te_level, fill_level)
        
        return SOMQualityReport(
            quantization_error=qe,
            topographic_error=te,
            fill_rate=fill_rate,
            qe_level=qe_level,
            te_level=te_level,
            fill_level=fill_level,
            overall_level=overall_level,
            total_units=total_units,
            active_units=active_units,
            empty_units=empty_units,
            diagnoses_per_unit_mean=mean_per_unit,
            diagnoses_per_unit_std=std_per_unit,
            diagnoses_per_unit_min=min_per_unit,
            diagnoses_per_unit_max=max_per_unit,
            orphan_diagnoses=orphans,
            grid_size=(h, w),
            total_diagnoses=total_diagnoses,
        )
    
    def _validate_som_model(self, som_model: Any) -> SOMQualityReport:
        """Валідація SOMModel"""
        # SOMModel має свої метрики
        qe = getattr(som_model, '_quantization_error', 0.0) or 0.0
        te = getattr(som_model, '_topographic_error', 0.0) or 0.0
        
        unit_to_diseases = getattr(som_model, '_unit_to_diseases', {})
        config = getattr(som_model, 'config', {})
        
        h = config.get('height', 15)
        w = config.get('width', 15)
        total_units = h * w
        
        # Статистика
        active_units = len([u for u, d in unit_to_diseases.items() if len(d) > 0])
        empty_units = total_units - active_units
        
        counts = [len(d) for d in unit_to_diseases.values() if len(d) > 0]
        if counts:
            mean_per_unit = np.mean(counts)
            std_per_unit = np.std(counts)
            min_per_unit = min(counts)
            max_per_unit = max(counts)
        else:
            mean_per_unit = std_per_unit = 0.0
            min_per_unit = max_per_unit = 0
        
        fill_rate = active_units / total_units if total_units > 0 else 0.0
        
        # Рівні якості
        qe_level = self._classify_qe(qe)
        te_level = self._classify_te(te)
        fill_level = self._classify_fill(fill_rate)
        overall_level = self._compute_overall(qe_level, te_level, fill_level)
        
        disease_names = getattr(som_model, '_disease_names', [])
        
        return SOMQualityReport(
            quantization_error=qe,
            topographic_error=te,
            fill_rate=fill_rate,
            qe_level=qe_level,
            te_level=te_level,
            fill_level=fill_level,
            overall_level=overall_level,
            total_units=total_units,
            active_units=active_units,
            empty_units=empty_units,
            diagnoses_per_unit_mean=mean_per_unit,
            diagnoses_per_unit_std=std_per_unit,
            diagnoses_per_unit_min=min_per_unit,
            diagnoses_per_unit_max=max_per_unit,
            orphan_diagnoses=[],
            grid_size=(h, w),
            total_diagnoses=len(disease_names),
        )
    
    def _classify_qe(self, qe: float) -> QualityLevel:
        """Класифікація QE"""
        if qe <= self.thresholds.qe_excellent:
            return QualityLevel.EXCELLENT
        elif qe <= self.thresholds.qe_good:
            return QualityLevel.GOOD
        elif qe <= self.thresholds.qe_acceptable:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    def _classify_te(self, te: float) -> QualityLevel:
        """Класифікація TE"""
        if te <= self.thresholds.te_excellent:
            return QualityLevel.EXCELLENT
        elif te <= self.thresholds.te_good:
            return QualityLevel.GOOD
        elif te <= self.thresholds.te_acceptable:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    def _classify_fill(self, fill_rate: float) -> QualityLevel:
        """Класифікація Fill Rate"""
        if self.thresholds.fill_min <= fill_rate <= self.thresholds.fill_max:
            if 0.7 <= fill_rate <= 0.9:
                return QualityLevel.EXCELLENT
            else:
                return QualityLevel.GOOD
        elif fill_rate >= self.thresholds.fill_min * 0.8:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    def _compute_overall(
        self,
        qe_level: QualityLevel,
        te_level: QualityLevel,
        fill_level: QualityLevel
    ) -> QualityLevel:
        """Обчислення загального рівня"""
        levels = [qe_level, te_level, fill_level]
        
        # Якщо є POOR — загальний POOR
        if QualityLevel.POOR in levels:
            return QualityLevel.POOR
        
        # Рахуємо
        excellent_count = levels.count(QualityLevel.EXCELLENT)
        good_count = levels.count(QualityLevel.GOOD)
        
        if excellent_count >= 2:
            return QualityLevel.EXCELLENT
        elif excellent_count + good_count >= 2:
            return QualityLevel.GOOD
        else:
            return QualityLevel.ACCEPTABLE
    
    def validate_from_checkpoint(
        self,
        checkpoint_path: str
    ) -> SOMQualityReport:
        """
        Валідація з файлу checkpoint.
        
        Args:
            checkpoint_path: Шлях до .pkl файлу
            
        Returns:
            SOMQualityReport
        """
        import pickle
        
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            som_model = data.get('som') or data.get('model')
            unit_to_diseases = data.get('unit_to_diseases', {})
            
            if som_model:
                # Якщо є збережені метрики — використовуємо їх
                qe = data.get('qe', 0.0)
                te = data.get('te', 0.0)
                
                # Створюємо звіт напряму
                h, w = som_model._weights.shape[:2]
                total_units = h * w
                
                active_units = len([u for u, d in unit_to_diseases.items() if len(d) > 0])
                empty_units = total_units - active_units
                
                counts = [len(d) for d in unit_to_diseases.values() if len(d) > 0]
                if counts:
                    mean_per_unit = np.mean(counts)
                    std_per_unit = np.std(counts)
                    min_per_unit = min(counts)
                    max_per_unit = max(counts)
                else:
                    mean_per_unit = std_per_unit = 0.0
                    min_per_unit = max_per_unit = 0
                
                fill_rate = active_units / total_units if total_units > 0 else 0.0
                
                qe_level = self._classify_qe(qe)
                te_level = self._classify_te(te)
                fill_level = self._classify_fill(fill_rate)
                overall_level = self._compute_overall(qe_level, te_level, fill_level)
                
                disease_names = data.get('disease_names', [])
                
                return SOMQualityReport(
                    quantization_error=qe,
                    topographic_error=te,
                    fill_rate=fill_rate,
                    qe_level=qe_level,
                    te_level=te_level,
                    fill_level=fill_level,
                    overall_level=overall_level,
                    total_units=total_units,
                    active_units=active_units,
                    empty_units=empty_units,
                    diagnoses_per_unit_mean=mean_per_unit,
                    diagnoses_per_unit_std=std_per_unit,
                    diagnoses_per_unit_min=min_per_unit,
                    diagnoses_per_unit_max=max_per_unit,
                    orphan_diagnoses=[],
                    grid_size=(h, w),
                    total_diagnoses=len(disease_names),
                )
        
        raise ValueError("Невалідний checkpoint")
