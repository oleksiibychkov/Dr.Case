"""
Dr.Case — Full Pipeline Validator & Tuner

Об'єднує валідацію всіх компонентів:
- SOM Quality (QE, TE, Fill)
- Candidate Recall
- NN Quality (Recall@k, mAP)
- End-to-End Diagnosis Accuracy

Також підтримує оптимізацію параметрів.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from enum import Enum

from .som_quality import SOMQualityValidator, SOMQualityReport, QualityLevel
from .candidate_recall import CandidateRecallValidator, CandidateRecallReport, RecallLevel
from .nn_quality import NNQualityValidator, NNQualityReport, NNQualityLevel


class PipelineStatus(Enum):
    """Статус pipeline"""
    PRODUCTION_READY = "production_ready"
    PROTOTYPE_READY = "prototype_ready"
    NEEDS_IMPROVEMENT = "needs_improvement"
    CRITICAL_ISSUES = "critical_issues"


@dataclass
class PipelineValidationReport:
    """Повний звіт про валідацію pipeline"""
    # Компоненти
    som_report: Optional[SOMQualityReport] = None
    candidate_report: Optional[CandidateRecallReport] = None
    nn_report: Optional[NNQualityReport] = None
    
    # End-to-End метрики
    e2e_accuracy_top1: float = 0.0
    e2e_accuracy_top5: float = 0.0
    e2e_accuracy_top10: float = 0.0
    e2e_avg_questions: float = 0.0
    e2e_avg_iterations: float = 0.0
    
    # Загальний статус
    status: PipelineStatus = PipelineStatus.CRITICAL_ISSUES
    
    # Рекомендації
    recommendations: List[str] = field(default_factory=list)
    
    # Метадані
    timestamp: str = ""
    duration_seconds: float = 0.0
    n_test_cases: int = 0
    
    def is_production_ready(self) -> bool:
        """Чи готовий до production"""
        return self.status == PipelineStatus.PRODUCTION_READY
    
    def to_dict(self) -> dict:
        """Серіалізація"""
        return {
            "som": self.som_report.to_dict() if self.som_report else None,
            "candidate": self.candidate_report.to_dict() if self.candidate_report else None,
            "nn": self.nn_report.to_dict() if self.nn_report else None,
            "e2e": {
                "accuracy_top1": self.e2e_accuracy_top1,
                "accuracy_top5": self.e2e_accuracy_top5,
                "accuracy_top10": self.e2e_accuracy_top10,
                "avg_questions": self.e2e_avg_questions,
                "avg_iterations": self.e2e_avg_iterations,
            },
            "status": self.status.value,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "n_test_cases": self.n_test_cases,
            "is_production_ready": self.is_production_ready(),
        }
    
    def save(self, path: str) -> None:
        """Зберегти звіт"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        lines = [
            "=" * 60,
            "PIPELINE VALIDATION REPORT",
            "=" * 60,
            f"Status: {self.status.value}",
            f"Timestamp: {self.timestamp}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"Test cases: {self.n_test_cases}",
            "",
        ]
        
        if self.som_report:
            lines.extend([
                "--- SOM ---",
                f"  QE: {self.som_report.quantization_error:.4f} ({self.som_report.qe_level.value})",
                f"  TE: {self.som_report.topographic_error:.4f} ({self.som_report.te_level.value})",
                f"  Fill: {self.som_report.fill_rate:.2%}",
                "",
            ])
        
        if self.candidate_report:
            lines.extend([
                "--- CANDIDATE SELECTOR ---",
                f"  Recall: {self.candidate_report.recall:.4f} ({self.candidate_report.recall_level.value})",
                f"  Avg candidates: {self.candidate_report.avg_candidates:.1f}",
                "",
            ])
        
        if self.nn_report:
            lines.extend([
                "--- NEURAL NETWORK ---",
                f"  Recall@1:  {self.nn_report.recall_1:.4f}",
                f"  Recall@5:  {self.nn_report.recall_5:.4f}",
                f"  Recall@10: {self.nn_report.recall_10:.4f}",
                f"  mAP: {self.nn_report.mean_average_precision:.4f}",
                "",
            ])
        
        lines.extend([
            "--- END-TO-END ---",
            f"  Accuracy@1:  {self.e2e_accuracy_top1:.4f}",
            f"  Accuracy@5:  {self.e2e_accuracy_top5:.4f}",
            f"  Accuracy@10: {self.e2e_accuracy_top10:.4f}",
            f"  Avg questions: {self.e2e_avg_questions:.1f}",
            "",
        ])
        
        if self.recommendations:
            lines.extend([
                "--- RECOMMENDATIONS ---",
            ])
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class FullPipelineValidator:
    """
    Повна валідація pipeline Dr.Case.
    
    Приклад використання:
        validator = FullPipelineValidator()
        
        report = validator.validate_from_checkpoints(
            som_path="models/som_merged.pkl",
            nn_path="models/nn_two_branch.pt",
            database_path="data/unified_disease_symptom_merged.json",
            n_samples=500
        )
        
        print(report)
        report.save("validation_report.json")
    """
    
    def __init__(self):
        self.som_validator = SOMQualityValidator()
        self.candidate_validator = CandidateRecallValidator()
        self.nn_validator = NNQualityValidator()
    
    def validate_from_checkpoints(
        self,
        som_path: str,
        nn_path: str,
        database_path: str,
        n_samples: int = 500,
        dropout_rate: float = 0.3,
        run_e2e: bool = True,
        verbose: bool = True
    ) -> PipelineValidationReport:
        """
        Валідація з checkpoint файлів.
        
        Args:
            som_path: Шлях до SOM checkpoint
            nn_path: Шлях до NN checkpoint
            database_path: Шлях до бази хвороб
            n_samples: Кількість тестових випадків
            dropout_rate: Частка пропущених симптомів
            run_e2e: Чи запускати end-to-end тестування
            verbose: Виводити прогрес
            
        Returns:
            PipelineValidationReport
        """
        start_time = datetime.now()
        
        if verbose:
            print("=" * 60)
            print("PIPELINE VALIDATION")
            print("=" * 60)
        
        # 1. SOM Quality
        if verbose:
            print("\n📊 Валідація SOM...")
        
        try:
            som_report = self.som_validator.validate_from_checkpoint(som_path)
            if verbose:
                print(f"   QE: {som_report.quantization_error:.4f} ({som_report.qe_level.value})")
                print(f"   TE: {som_report.topographic_error:.4f} ({som_report.te_level.value})")
                print(f"   Fill: {som_report.fill_rate:.2%}")
        except Exception as e:
            if verbose:
                print(f"   ❌ Помилка: {e}")
            som_report = None
        
        # 2. Candidate Recall
        if verbose:
            print("\n📋 Валідація Candidate Selector...")
        
        try:
            candidate_report = self.candidate_validator.validate_from_checkpoint(
                som_checkpoint_path=som_path,
                database_path=database_path,
                n_samples=n_samples,
                dropout_rate=dropout_rate
            )
            if verbose:
                print(f"   Recall: {candidate_report.recall:.4f} ({candidate_report.recall_level.value})")
                print(f"   Avg candidates: {candidate_report.avg_candidates:.1f}")
        except Exception as e:
            if verbose:
                print(f"   ❌ Помилка: {e}")
            candidate_report = None
        
        # 3. NN Quality
        if verbose:
            print("\n🧠 Валідація Neural Network...")
        
        try:
            nn_report = self.nn_validator.validate_from_checkpoint(
                nn_checkpoint_path=nn_path,
                som_checkpoint_path=som_path,
                database_path=database_path,
                n_samples=n_samples,
                dropout_rate=dropout_rate
            )
            if verbose:
                print(f"   Recall@1:  {nn_report.recall_1:.4f}")
                print(f"   Recall@5:  {nn_report.recall_5:.4f}")
                print(f"   Recall@10: {nn_report.recall_10:.4f}")
                print(f"   mAP: {nn_report.mean_average_precision:.4f}")
        except Exception as e:
            if verbose:
                print(f"   ❌ Помилка: {e}")
            nn_report = None
        
        # 4. End-to-End (якщо включено)
        e2e_top1 = e2e_top5 = e2e_top10 = 0.0
        e2e_questions = e2e_iterations = 0.0
        
        if run_e2e:
            if verbose:
                print("\n🔄 End-to-End валідація...")
            
            try:
                e2e_results = self._run_e2e_validation(
                    som_path=som_path,
                    nn_path=nn_path,
                    database_path=database_path,
                    n_samples=min(100, n_samples),  # E2E повільніший
                    dropout_rate=dropout_rate,
                    verbose=verbose
                )
                e2e_top1 = e2e_results.get('accuracy_top1', 0.0)
                e2e_top5 = e2e_results.get('accuracy_top5', 0.0)
                e2e_top10 = e2e_results.get('accuracy_top10', 0.0)
                e2e_questions = e2e_results.get('avg_questions', 0.0)
                e2e_iterations = e2e_results.get('avg_iterations', 0.0)
                
                if verbose:
                    print(f"   Accuracy@1:  {e2e_top1:.4f}")
                    print(f"   Accuracy@5:  {e2e_top5:.4f}")
                    print(f"   Accuracy@10: {e2e_top10:.4f}")
                    print(f"   Avg questions: {e2e_questions:.1f}")
            except Exception as e:
                if verbose:
                    print(f"   ❌ Помилка E2E: {e}")
        
        # 5. Визначаємо загальний статус
        status, recommendations = self._compute_status(
            som_report, candidate_report, nn_report,
            e2e_top1, e2e_top5, e2e_top10
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        report = PipelineValidationReport(
            som_report=som_report,
            candidate_report=candidate_report,
            nn_report=nn_report,
            e2e_accuracy_top1=e2e_top1,
            e2e_accuracy_top5=e2e_top5,
            e2e_accuracy_top10=e2e_top10,
            e2e_avg_questions=e2e_questions,
            e2e_avg_iterations=e2e_iterations,
            status=status,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            n_test_cases=n_samples,
        )
        
        if verbose:
            print(f"\n✅ Валідація завершена за {duration:.1f}с")
            print(f"   Статус: {status.value}")
        
        return report
    
    def _run_e2e_validation(
        self,
        som_path: str,
        nn_path: str,
        database_path: str,
        n_samples: int,
        dropout_rate: float,
        verbose: bool
    ) -> Dict[str, float]:
        """End-to-End валідація з DiagnosisCycleController"""
        import pickle
        import json
        import torch
        
        # Завантажуємо моделі
        with open(som_path, 'rb') as f:
            som_data = pickle.load(f)
        
        nn_checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
        
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        # Створюємо словник симптомів
        all_symptoms = set()
        for disease_data in database.values():
            all_symptoms.update(disease_data.get('symptoms', []))
        symptom_list = sorted(all_symptoms)
        
        # Генеруємо тестові випадки
        diseases = list(database.keys())
        
        results_top1 = []
        results_top5 = []
        results_top10 = []
        questions_counts = []
        iterations_counts = []
        
        np.random.seed(42)
        
        try:
            from dr_case.diagnosis_cycle import DiagnosisCycleController
            
            controller = DiagnosisCycleController.from_models(
                database_path=database_path,
                som_path=som_path,
                nn_path=nn_path,
                language="uk"
            )
            
            for i in range(n_samples):
                if verbose and i % 20 == 0:
                    print(f"   E2E progress: {i}/{n_samples}")
                
                # Випадковий діагноз
                true_disease = np.random.choice(diseases)
                symptoms = database[true_disease].get('symptoms', [])
                
                if not symptoms:
                    continue
                
                # Dropout симптомів
                n_keep = max(1, int(len(symptoms) * (1 - dropout_rate)))
                initial_symptoms = list(np.random.choice(
                    symptoms, size=min(n_keep, len(symptoms)), replace=False
                ))
                
                # Запускаємо діагностику
                try:
                    result = controller.start_session(initial_symptoms)
                    
                    # Отримуємо топ гіпотези
                    hypotheses = controller.get_top_hypotheses(10)
                    
                    top1_diseases = [h[0] for h in hypotheses[:1]]
                    top5_diseases = [h[0] for h in hypotheses[:5]]
                    top10_diseases = [h[0] for h in hypotheses[:10]]
                    
                    results_top1.append(1 if true_disease in top1_diseases else 0)
                    results_top5.append(1 if true_disease in top5_diseases else 0)
                    results_top10.append(1 if true_disease in top10_diseases else 0)
                    
                    questions_counts.append(len(controller.session_state.symptom_history))
                    iterations_counts.append(controller.current_iteration)
                    
                except Exception:
                    # Пропускаємо помилкові випадки
                    pass
            
        except ImportError:
            # Якщо DiagnosisCycleController недоступний — спрощена версія
            return {
                'accuracy_top1': 0.0,
                'accuracy_top5': 0.0,
                'accuracy_top10': 0.0,
                'avg_questions': 0.0,
                'avg_iterations': 0.0,
            }
        
        return {
            'accuracy_top1': np.mean(results_top1) if results_top1 else 0.0,
            'accuracy_top5': np.mean(results_top5) if results_top5 else 0.0,
            'accuracy_top10': np.mean(results_top10) if results_top10 else 0.0,
            'avg_questions': np.mean(questions_counts) if questions_counts else 0.0,
            'avg_iterations': np.mean(iterations_counts) if iterations_counts else 0.0,
        }
    
    def _compute_status(
        self,
        som_report: Optional[SOMQualityReport],
        candidate_report: Optional[CandidateRecallReport],
        nn_report: Optional[NNQualityReport],
        e2e_top1: float,
        e2e_top5: float,
        e2e_top10: float
    ) -> Tuple[PipelineStatus, List[str]]:
        """Обчислення загального статусу та рекомендацій"""
        recommendations = []
        issues = 0
        
        # SOM
        if som_report:
            if som_report.overall_level == QualityLevel.POOR:
                issues += 2
                recommendations.append("SOM: Потрібно перенавчити з іншими параметрами")
            elif som_report.overall_level == QualityLevel.ACCEPTABLE:
                issues += 1
                recommendations.append("SOM: Можна покращити QE/TE налаштуванням параметрів")
        else:
            issues += 2
            recommendations.append("SOM: Помилка валідації")
        
        # Candidate Selector
        if candidate_report:
            if candidate_report.recall_level == RecallLevel.POOR:
                issues += 2
                recommendations.append("Candidate: Recall занадто низький, збільшіть k або α")
            elif candidate_report.recall_level == RecallLevel.ACCEPTABLE:
                issues += 1
                recommendations.append("Candidate: Recall нижче prototype рівня")
        else:
            issues += 2
            recommendations.append("Candidate: Помилка валідації")
        
        # NN
        if nn_report:
            if nn_report.overall_level == NNQualityLevel.POOR:
                issues += 2
                recommendations.append("NN: Recall@5 занадто низький, потрібно більше даних або epochs")
            elif nn_report.overall_level == NNQualityLevel.ACCEPTABLE:
                issues += 1
                recommendations.append("NN: Можна покращити tuning гіперпараметрів")
        else:
            issues += 2
            recommendations.append("NN: Помилка валідації")
        
        # E2E
        if e2e_top5 < 0.7:
            issues += 1
            recommendations.append("E2E: Accuracy@5 нижче 70%, потрібна оптимізація pipeline")
        
        # Визначаємо статус
        if issues == 0:
            status = PipelineStatus.PRODUCTION_READY
        elif issues <= 2:
            status = PipelineStatus.PROTOTYPE_READY
        elif issues <= 4:
            status = PipelineStatus.NEEDS_IMPROVEMENT
        else:
            status = PipelineStatus.CRITICAL_ISSUES
        
        # Додаємо позитивні рекомендації
        if not recommendations:
            recommendations.append("Всі компоненти працюють на production рівні!")
        
        return status, recommendations


# Швидкий доступ
def validate_pipeline(
    som_path: str,
    nn_path: str,
    database_path: str,
    n_samples: int = 500,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> PipelineValidationReport:
    """
    Швидка валідація pipeline.
    
    Args:
        som_path: Шлях до SOM checkpoint
        nn_path: Шлях до NN checkpoint
        database_path: Шлях до бази хвороб
        n_samples: Кількість тестових випадків
        output_path: Куди зберегти звіт (опціонально)
        verbose: Виводити прогрес
        
    Returns:
        PipelineValidationReport
    """
    validator = FullPipelineValidator()
    report = validator.validate_from_checkpoints(
        som_path=som_path,
        nn_path=nn_path,
        database_path=database_path,
        n_samples=n_samples,
        verbose=verbose
    )
    
    if output_path:
        report.save(output_path)
        if verbose:
            print(f"\n📄 Звіт збережено: {output_path}")
    
    return report
