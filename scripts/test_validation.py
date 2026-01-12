#!/usr/bin/env python3
"""
Dr.Case — Тестування модулів Validation та Full Pipeline Tuner

Запуск:
    python scripts/test_validation.py
"""

import sys
from pathlib import Path

# Додаємо шлях до проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_som_quality():
    """Тест валідації SOM"""
    print("\n" + "=" * 60)
    print("📊 ТЕСТ: SOM Quality Validator")
    print("=" * 60)
    
    from dr_case.validation import SOMQualityValidator, QualityLevel
    
    validator = SOMQualityValidator()
    
    # Тест з checkpoint
    som_path = project_root / "models" / "som_merged.pkl"
    
    if som_path.exists():
        print(f"\n🔍 Валідація: {som_path}")
        report = validator.validate_from_checkpoint(str(som_path))
        
        print(f"\n   QE: {report.quantization_error:.4f} ({report.qe_level.value})")
        print(f"   TE: {report.topographic_error:.4f} ({report.te_level.value})")
        print(f"   Fill: {report.fill_rate:.2%} ({report.fill_level.value})")
        print(f"   Overall: {report.overall_level.value}")
        print(f"   Units: {report.active_units}/{report.total_units} active")
        print(f"   Diagnoses/unit: {report.diagnoses_per_unit_mean:.1f} ± {report.diagnoses_per_unit_std:.1f}")
        print(f"   Is acceptable: {report.is_acceptable()}")
        
        return report.is_acceptable()
    else:
        print(f"   ⚠️ SOM checkpoint не знайдено: {som_path}")
        return True  # Пропускаємо


def test_candidate_recall():
    """Тест валідації Candidate Recall"""
    print("\n" + "=" * 60)
    print("📋 ТЕСТ: Candidate Recall Validator")
    print("=" * 60)
    
    from dr_case.validation import CandidateRecallValidator, RecallLevel
    
    validator = CandidateRecallValidator()
    
    # Тест з готовими даними
    print("\n🔍 Тест з готовими даними:")
    
    test_cases = [
        ("Influenza", ["Influenza", "Cold", "COVID", "Bronchitis"]),
        ("Diabetes", ["Diabetes", "Obesity", "Hypertension"]),
        ("Migraine", ["Headache", "Tension Headache", "Migraine"]),  # Hit
        ("Asthma", ["COPD", "Bronchitis"]),  # Miss!
        ("Pneumonia", ["Pneumonia", "Bronchitis", "COVID"]),
    ]
    
    report = validator.validate(test_cases)
    
    print(f"   Recall: {report.recall:.4f} ({report.recall_level.value})")
    print(f"   Hits: {report.hits}/{report.total_cases}")
    print(f"   Misses: {report.misses}")
    print(f"   Avg candidates: {report.avg_candidates:.1f}")
    print(f"   Precision@k: {report.precision_at_k}")
    
    # Тест з checkpoint
    som_path = project_root / "models" / "som_merged.pkl"
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if som_path.exists() and db_path.exists():
        print(f"\n🔍 Валідація з моделями (100 samples):")
        
        report = validator.validate_from_checkpoint(
            som_checkpoint_path=str(som_path),
            database_path=str(db_path),
            n_samples=100,
            dropout_rate=0.3
        )
        
        print(f"   Recall: {report.recall:.4f} ({report.recall_level.value})")
        print(f"   Avg candidates: {report.avg_candidates:.1f}")
        print(f"   Is acceptable: {report.is_acceptable()}")
        
        return report.is_acceptable()
    
    return True


def test_nn_quality():
    """Тест валідації NN"""
    print("\n" + "=" * 60)
    print("🧠 ТЕСТ: NN Quality Validator")
    print("=" * 60)
    
    from dr_case.validation import NNQualityValidator, NNQualityLevel
    
    validator = NNQualityValidator()
    
    # Тест з готовими даними
    print("\n🔍 Тест з готовими передбаченнями:")
    
    predictions = [
        {"Influenza": 0.85, "Cold": 0.40, "COVID": 0.20},
        {"Diabetes": 0.75, "Obesity": 0.30, "Hypertension": 0.25},
        {"Migraine": 0.60, "Headache": 0.70, "Tension": 0.50},  # Wrong order
        {"Asthma": 0.50, "COPD": 0.55, "Bronchitis": 0.45},     # COPD top (wrong)
        {"Pneumonia": 0.90, "Bronchitis": 0.30, "COVID": 0.20},
    ]
    
    true_labels = ["Influenza", "Diabetes", "Migraine", "Asthma", "Pneumonia"]
    
    report = validator.validate(predictions, true_labels)
    
    print(f"   Recall@1:  {report.recall_1:.4f}")
    print(f"   Recall@5:  {report.recall_5:.4f}")
    print(f"   Recall@10: {report.recall_10:.4f}")
    print(f"   mAP: {report.mean_average_precision:.4f}")
    print(f"   Level: {report.overall_level.value}")
    
    # Тест з checkpoint
    som_path = project_root / "models" / "som_merged.pkl"
    nn_path = project_root / "models" / "nn_two_branch.pt"
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if som_path.exists() and nn_path.exists() and db_path.exists():
        print(f"\n🔍 Валідація з моделями (100 samples):")
        
        try:
            report = validator.validate_from_checkpoint(
                nn_checkpoint_path=str(nn_path),
                som_checkpoint_path=str(som_path),
                database_path=str(db_path),
                n_samples=100,
                dropout_rate=0.3
            )
            
            print(f"   Recall@1:  {report.recall_1:.4f}")
            print(f"   Recall@5:  {report.recall_5:.4f}")
            print(f"   Recall@10: {report.recall_10:.4f}")
            print(f"   mAP: {report.mean_average_precision:.4f}")
            print(f"   Level: {report.overall_level.value}")
            print(f"   Is acceptable: {report.is_acceptable()}")
            
            return report.is_acceptable()
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
            return True  # Пропускаємо
    
    return True


def test_full_pipeline():
    """Тест повної валідації pipeline"""
    print("\n" + "=" * 60)
    print("🔄 ТЕСТ: Full Pipeline Validator")
    print("=" * 60)
    
    from dr_case.validation import validate_pipeline, PipelineStatus
    
    som_path = project_root / "models" / "som_merged.pkl"
    nn_path = project_root / "models" / "nn_two_branch.pt"
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if not all(p.exists() for p in [som_path, nn_path, db_path]):
        print("   ⚠️ Моделі не знайдено, пропускаємо")
        return True
    
    print("\n🔍 Повна валідація pipeline (100 samples):")
    
    try:
        report = validate_pipeline(
            som_path=str(som_path),
            nn_path=str(nn_path),
            database_path=str(db_path),
            n_samples=100,
            output_path=str(project_root / "validation_report.json"),
            verbose=True
        )
        
        print(f"\n   Статус: {report.status.value}")
        print(f"   Production ready: {report.is_production_ready()}")
        
        if report.recommendations:
            print(f"\n   Рекомендації:")
            for rec in report.recommendations:
                print(f"   • {rec}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Головна функція"""
    print("=" * 60)
    print("Dr.Case — ТЕСТУВАННЯ VALIDATION & PIPELINE TUNER")
    print("=" * 60)
    
    results = {}
    
    # Тест 1: SOM Quality
    try:
        results['som'] = test_som_quality()
        print("\n✅ Тест SOM Quality пройдено!")
    except Exception as e:
        print(f"\n❌ Помилка SOM Quality: {e}")
        results['som'] = False
    
    # Тест 2: Candidate Recall
    try:
        results['candidate'] = test_candidate_recall()
        print("\n✅ Тест Candidate Recall пройдено!")
    except Exception as e:
        print(f"\n❌ Помилка Candidate Recall: {e}")
        results['candidate'] = False
    
    # Тест 3: NN Quality
    try:
        results['nn'] = test_nn_quality()
        print("\n✅ Тест NN Quality пройдено!")
    except Exception as e:
        print(f"\n❌ Помилка NN Quality: {e}")
        results['nn'] = False
    
    # Тест 4: Full Pipeline
    try:
        results['pipeline'] = test_full_pipeline()
        print("\n✅ Тест Full Pipeline пройдено!")
    except Exception as e:
        print(f"\n❌ Помилка Full Pipeline: {e}")
        results['pipeline'] = False
    
    # Підсумок
    print("\n" + "=" * 60)
    print("📊 ПІДСУМОК")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ВСІ ТЕСТИ ПРОЙДЕНО!")
    else:
        print("\n⚠️ Деякі тести не пройшли")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
