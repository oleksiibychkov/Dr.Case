"""
Dr.Case — Налаштування системи

Всі параметри системи зібрані в dataclass-и для:
- Типізації та валідації
- Легкого доступу через config.som.grid_size
- Серіалізації в YAML/JSON
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class SOMInitialization(str, Enum):
    """Метод ініціалізації SOM"""
    PCA = "pca"
    RANDOM = "random"


class NeighborhoodFunction(str, Enum):
    """Функція сусідства SOM"""
    GAUSSIAN = "gaussian"
    BUBBLE = "bubble"


class SelectorPolicy(str, Enum):
    """Політика відбору кандидатів"""
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    CUMULATIVE_MASS = "cumulative_mass"
    COMBINED = "combined"


class NoiseMethod(str, Enum):
    """Метод додавання шуму"""
    UNIFORM = "uniform"
    FREQUENCY_WEIGHTED = "frequency_weighted"
    SOM_LOCAL = "som_local"


# =============================================================================
# SOM CONFIGURATION
# =============================================================================

@dataclass
class SOMConfig:
    """Параметри Self-Organizing Map"""
    
    # Топологія
    grid_height: int = 15
    grid_width: int = 15
    input_dim: int = 461  # кількість симптомів
    
    # Навчання
    epochs: int = 1000
    learning_rate_init: float = 0.5
    learning_rate_final: float = 0.01
    sigma_init: float = 7.5  # max(H, W) / 2
    sigma_final: float = 0.5
    
    # Методи
    initialization: SOMInitialization = SOMInitialization.PCA
    neighborhood_function: NeighborhoodFunction = NeighborhoodFunction.GAUSSIAN
    
    # Early stopping
    early_stopping_patience: int = 50
    
    # Пороги якості
    qe_acceptable: float = 0.5
    qe_good: float = 0.3
    te_acceptable: float = 0.2
    te_good: float = 0.1
    fill_min: float = 0.6
    fill_max: float = 0.95
    
    @classmethod
    def for_disease_count(cls, n_diseases: int, n_symptoms: int) -> "SOMConfig":
        """Створити конфігурацію на основі кількості діагнозів (формула Vesanto)"""
        import math
        grid_size = int(5 * math.sqrt(n_diseases))
        grid_size = max(10, min(grid_size, 50))  # обмеження 10-50
        
        return cls(
            grid_height=grid_size,
            grid_width=grid_size,
            input_dim=n_symptoms,
            sigma_init=grid_size / 2
        )


# =============================================================================
# CANDIDATE SELECTOR CONFIGURATION
# =============================================================================

@dataclass
class CandidateSelectorConfig:
    """Параметри відбору кандидатів"""
    
    # Основні параметри
    alpha: float = 0.9          # cumulative mass (90%)
    k: int = 6                  # max юнітів
    tau: float = 0.01           # мінімальний поріг membership
    
    # Політика
    policy: SelectorPolicy = SelectorPolicy.COMBINED
    
    # Гарантії
    target_recall: float = 0.995
    
    # Membership
    membership_lambda: float = 1.0  # параметр "гостроти" softmax


# =============================================================================
# PSEUDO GENERATION CONFIGURATION
# =============================================================================

@dataclass
class DropoutConfig:
    """Параметри dropout симптомів"""
    initial_min: float = 0.5
    initial_max: float = 0.7
    after_questions_min: float = 0.2
    after_questions_max: float = 0.4
    full_examination_min: float = 0.0
    full_examination_max: float = 0.1


@dataclass
class NoiseConfig:
    """Параметри шуму"""
    # Розподіл кількості зайвих симптомів
    distribution: Dict[int, float] = field(default_factory=lambda: {
        0: 0.5, 1: 0.25, 2: 0.15, 3: 0.10
    })
    method: NoiseMethod = NoiseMethod.SOM_LOCAL


@dataclass
class ComorbidityConfig:
    """Параметри коморбідності"""
    single: float = 0.70      # 1 діагноз
    double: float = 0.25      # 2 діагнози
    triple_plus: float = 0.05 # 3+ діагнози


@dataclass
class PseudoGenerationConfig:
    """Параметри генерації псевдопацієнтів"""
    
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    comorbidity: ComorbidityConfig = field(default_factory=ComorbidityConfig)
    
    # Кількість
    pseudo_per_diagnosis_min: int = 10
    pseudo_per_diagnosis_max: int = 50
    pseudo_per_diagnosis_default: int = 25
    
    # Розподіл типів
    distribution: Dict[str, float] = field(default_factory=lambda: {
        "single_high_dropout": 0.30,
        "single_moderate_dropout": 0.25,
        "single_with_noise": 0.15,
        "multi_label_2": 0.20,
        "multi_label_3": 0.05,
        "iterative_sequences": 0.05,
    })


# =============================================================================
# NEURAL NETWORK CONFIGURATION
# =============================================================================

@dataclass
class NNArchitectureConfig:
    """Архітектура Multilabel NN"""
    
    # Розмірності
    input_dim_symptoms: int = 461
    input_dim_som: int = 6
    output_dim: int = 842
    
    # Branch симптомів
    branch_symptoms_layers: List[int] = field(default_factory=lambda: [256, 128])
    branch_symptoms_dropout: List[float] = field(default_factory=lambda: [0.3, 0.3])
    
    # Branch SOM
    branch_som_layers: List[int] = field(default_factory=lambda: [64, 32])
    branch_som_dropout: List[float] = field(default_factory=lambda: [0.2, 0.2])
    
    # Combined
    combined_layers: List[int] = field(default_factory=lambda: [128])
    combined_dropout: List[float] = field(default_factory=lambda: [0.3])
    
    # Загальні
    activation: str = "relu"
    output_activation: str = "sigmoid"
    use_batchnorm: bool = True


@dataclass
class NNTrainingConfig:
    """Параметри навчання NN"""
    
    # Оптимізатор
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # LR Schedule
    scheduler: str = "cosine_warmup"
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    
    # Batch size (залежить від розміру датасету)
    batch_size_small: int = 32      # N < 5000
    batch_size_medium: int = 64     # 5000-50000
    batch_size_large: int = 256     # N > 50000
    
    # Epochs
    max_epochs_small: int = 200
    max_epochs_medium: int = 100
    max_epochs_large: int = 50
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Регуляризація
    label_smoothing: float = 0.05
    gradient_clip: float = 1.0
    
    # Data split
    train_ratio: float = 0.75
    validation_ratio: float = 0.15
    test_ratio: float = 0.10
    
    def get_batch_size(self, dataset_size: int) -> int:
        """Отримати batch size залежно від розміру датасету"""
        if dataset_size < 5000:
            return self.batch_size_small
        elif dataset_size < 50000:
            return self.batch_size_medium
        else:
            return self.batch_size_large
    
    def get_max_epochs(self, dataset_size: int) -> int:
        """Отримати max epochs залежно від розміру датасету"""
        if dataset_size < 5000:
            return self.max_epochs_small
        elif dataset_size < 50000:
            return self.max_epochs_medium
        else:
            return self.max_epochs_large


# =============================================================================
# QUESTION ENGINE CONFIGURATION
# =============================================================================

@dataclass
class QuestionEngineConfig:
    """Параметри механізму питань"""
    
    max_questions_per_iteration: int = 1
    max_total_questions: int = 20
    min_information_gain: float = 0.01
    
    # Пріоритети
    prefer_easy_questions: bool = True
    prefer_discriminative: bool = True


# =============================================================================
# STOPPING CRITERIA CONFIGURATION
# =============================================================================

@dataclass
class DominanceConfig:
    """Критерій DOMINANCE"""
    threshold: float = 0.85
    gap: float = 0.3


@dataclass
class StabilityConfig:
    """Критерій STABILITY"""
    iterations: int = 3
    tolerance: float = 0.05


@dataclass
class NeedTestConfig:
    """Критерій NEED_TEST"""
    top_n_similar: int = 2
    similarity_threshold: float = 0.1


@dataclass
class SafetyConfig:
    """Критерій SAFETY"""
    max_iterations: int = 20
    timeout_minutes: int = 30


@dataclass
class StoppingCriteriaConfig:
    """Всі критерії зупинки"""
    dominance: DominanceConfig = field(default_factory=DominanceConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    need_test: NeedTestConfig = field(default_factory=NeedTestConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

@dataclass
class QualityThresholds:
    """Пороги якості системи"""
    
    # Candidate Selector
    candidate_recall_prototype: float = 0.99
    candidate_recall_production: float = 0.995
    
    # Multilabel NN
    recall_at_1_prototype: float = 0.50
    recall_at_1_production: float = 0.70
    recall_at_5_prototype: float = 0.85
    recall_at_5_production: float = 0.95
    recall_at_10_prototype: float = 0.92
    recall_at_10_production: float = 0.99
    map_prototype: float = 0.60
    map_production: float = 0.80


# =============================================================================
# FEEDBACK CONFIGURATION
# =============================================================================

@dataclass
class TreatmentFailureConfig:
    """Параметри обробки невдачі лікування"""
    downgrade_factor: float = 0.5
    boost_alternatives_factor: float = 1.2
    expand_alpha: float = 0.95
    expand_k: int = 10


@dataclass
class FeedbackConfig:
    """Параметри зворотного зв'язку"""
    treatment_failure: TreatmentFailureConfig = field(default_factory=TreatmentFailureConfig)


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

@dataclass
class DrCaseConfig:
    """
    Головна конфігурація Dr.Case
    
    Об'єднує всі параметри системи в одному місці.
    
    Приклад використання:
        config = DrCaseConfig()
        print(config.som.grid_height)  # 15
        print(config.nn_training.learning_rate)  # 0.001
    """
    
    # Метадані
    version: str = "1.0.0"
    project_name: str = "Dr.Case"
    
    # Дані
    n_diseases: int = 842
    n_symptoms: int = 461
    
    # Компоненти
    som: SOMConfig = field(default_factory=SOMConfig)
    candidate_selector: CandidateSelectorConfig = field(default_factory=CandidateSelectorConfig)
    pseudo_generation: PseudoGenerationConfig = field(default_factory=PseudoGenerationConfig)
    nn_architecture: NNArchitectureConfig = field(default_factory=NNArchitectureConfig)
    nn_training: NNTrainingConfig = field(default_factory=NNTrainingConfig)
    question_engine: QuestionEngineConfig = field(default_factory=QuestionEngineConfig)
    stopping_criteria: StoppingCriteriaConfig = field(default_factory=StoppingCriteriaConfig)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    
    # Шляхи (відносні)
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    @classmethod
    def for_database(cls, n_diseases: int, n_symptoms: int) -> "DrCaseConfig":
        """Створити конфігурацію для конкретної бази даних"""
        config = cls(
            n_diseases=n_diseases,
            n_symptoms=n_symptoms,
        )
        
        # Оновити SOM для кількості діагнозів
        config.som = SOMConfig.for_disease_count(n_diseases, n_symptoms)
        
        # Оновити NN dimensions
        config.nn_architecture.input_dim_symptoms = n_symptoms
        config.nn_architecture.output_dim = n_diseases
        
        return config


# =============================================================================
# DEFAULT CONFIG INSTANCE
# =============================================================================

def get_default_config() -> DrCaseConfig:
    """Отримати конфігурацію за замовчуванням для бази 842 діагнози / 461 симптом"""
    return DrCaseConfig.for_database(n_diseases=842, n_symptoms=461)
