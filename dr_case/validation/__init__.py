"""
Dr.Case — Validation Module

Метрики якості всіх компонентів системи:
- SOM Quality (QE, TE, Fill Rate)
- Candidate Recall
- NN Quality (Recall@k, mAP)
- Full Pipeline Validation

Приклад використання:
    from dr_case.validation import validate_pipeline
    
    report = validate_pipeline(
        som_path="models/som_merged.pkl",
        nn_path="models/nn_two_branch.pt",
        database_path="data/unified_disease_symptom_merged.json",
        n_samples=500,
        output_path="validation_report.json"
    )
    
    print(report)
    
    if report.is_production_ready():
        print("Pipeline готовий до production!")
"""

from .som_quality import (
    SOMQualityValidator,
    SOMQualityReport,
    SOMQualityThresholds,
    QualityLevel,
)

from .candidate_recall import (
    CandidateRecallValidator,
    CandidateRecallReport,
    CandidateRecallThresholds,
    RecallLevel,
)

from .nn_quality import (
    NNQualityValidator,
    NNQualityReport,
    NNQualityThresholds,
    NNQualityLevel,
)

from .pipeline_validator import (
    FullPipelineValidator,
    PipelineValidationReport,
    PipelineStatus,
    validate_pipeline,
)


__all__ = [
    # SOM
    "SOMQualityValidator",
    "SOMQualityReport",
    "SOMQualityThresholds",
    "QualityLevel",
    
    # Candidate
    "CandidateRecallValidator",
    "CandidateRecallReport",
    "CandidateRecallThresholds",
    "RecallLevel",
    
    # NN
    "NNQualityValidator",
    "NNQualityReport",
    "NNQualityThresholds",
    "NNQualityLevel",
    
    # Pipeline
    "FullPipelineValidator",
    "PipelineValidationReport",
    "PipelineStatus",
    "validate_pipeline",
]
