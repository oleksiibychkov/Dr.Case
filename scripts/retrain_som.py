"""
Dr.Case — Перенавчання SOM на очищених даних (844 хвороби)

Запуск:
    python scripts/retrain_som.py
"""

import sys
import os
from pathlib import Path

# Додаємо корінь проекту до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dr_case.config import SOMConfig
from dr_case.som import SOMTrainer


def main():
    print("=" * 70)
    print("Dr.Case — ПЕРЕНАВЧАННЯ SOM")
    print("=" * 70)
    
    # Шляхи
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    output_model_path = project_root / "models" / "som_merged.pkl"
    output_tuning_path = project_root / "models" / "som_merged_tuning_result.json"
    
    # Перевіряємо наявність даних
    if not database_path.exists():
        print(f"❌ Файл не знайдено: {database_path}")
        return
    
    print(f"\n📁 База даних: {database_path}")
    print(f"📁 Вихідна модель: {output_model_path}")
    
    # Конфігурація SOM (оптимізована для 844 хвороб)
    # Формула Vesanto: grid_size ≈ 5 * sqrt(N) = 5 * sqrt(844) ≈ 145 → 12x12 або 15x15
    config = SOMConfig(
        grid_height=30,          # Збільшуємо для кращого розподілу
        grid_width=30,
        input_dim=460,           # Кількість симптомів (оновлено після очищення)
        epochs=500,
        learning_rate_init=0.5,
        learning_rate_final=0.01,
        sigma_init=15.0,         # Початковий радіус = grid/2
        sigma_final=0.5,
    )
    
    print(f"\n⚙️ Конфігурація SOM:")
    print(f"   Grid: {config.grid_height}x{config.grid_width}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Learning rate: {config.learning_rate_init} → {config.learning_rate_final}")
    print(f"   Sigma: {config.sigma_init} → {config.sigma_final}")
    
    # Створюємо тренер
    trainer = SOMTrainer(config)
    
    # Навчаємо
    print("\n🔄 Навчання SOM...")
    print("-" * 70)
    
    som_model, metrics = trainer.train_from_database(
        str(database_path),
        epochs=config.epochs,
        verbose=True
    )
    
    print("-" * 70)
    print(f"\n📊 Результати:")
    print(f"   QE (Quantization Error): {metrics['qe']:.4f}")
    print(f"   TE (Topographic Error): {metrics['te']:.4f}")
    print(f"   Fill ratio: {metrics['fill_ratio']:.2%}")
    print(f"   Filled units: {metrics['filled_units']}/{metrics['total_units']}")
    
    # Оцінюємо Candidate Recall
    print("\n📈 Оцінка Candidate Recall...")
    
    # Діагностика
    print(f"   DEBUG: filled_units = {len(trainer.som_model.filled_units)}")
    print(f"   DEBUG: unit_to_diseases = {len(trainer.som_model._unit_to_diseases)}")
    print(f"   DEBUG: disease_matrix shape = {trainer.disease_matrix.shape}")
    print(f"   DEBUG: disease_names count = {len(trainer.disease_names)}")
    
    # Тест одного вектора
    from dr_case.config import CandidateSelectorConfig
    from dr_case.som.projector import SOMProjector
    
    test_config = CandidateSelectorConfig(alpha=0.95, k=10, tau=0.001)
    test_projector = SOMProjector(trainer.som_model, test_config)
    test_result = test_projector.project(trainer.disease_matrix[0])
    print(f"   DEBUG: test projection - active_units={len(test_result.active_units)}, candidates={len(test_result.candidate_diseases)}")
    
    recall_metrics = trainer.evaluate_candidate_recall(alpha=0.95, k=10, tau=0.001)
    print(f"   Recall: {recall_metrics['recall']:.4f} ({recall_metrics['hits']}/{recall_metrics['total']})")
    print(f"   Avg candidates: {recall_metrics['avg_candidates']:.1f}")
    print(f"   Min/Max candidates: {recall_metrics['min_candidates']}/{recall_metrics['max_candidates']}")
    
    # Зберігаємо модель
    print(f"\n💾 Збереження моделі...")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_model_path))
    
    # Зберігаємо результати тюнінгу
    import json
    
    # Конвертуємо numpy типи в Python типи
    def to_python(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python(v) for v in obj]
        return obj
    
    tuning_result = {
        "database": str(database_path),
        "best_params": {
            "grid_height": config.grid_height,
            "grid_width": config.grid_width,
            "epochs": config.epochs,
            "learning_rate_init": config.learning_rate_init,
            "sigma_init": config.sigma_init,
        },
        "selector_params": {
            "alpha": 0.95,
            "k": 10,
            "tau": 0.001,
        },
        "best_metrics": {
            "qe": float(metrics['qe']),
            "te": float(metrics['te']),
            "fill_ratio": float(metrics['fill_ratio']),
        },
        "recall_metrics": to_python(recall_metrics),
    }
    
    with open(output_tuning_path, 'w') as f:
        json.dump(tuning_result, f, indent=2)
    print(f"   Tuning result: {output_tuning_path}")
    
    print("\n" + "=" * 70)
    print("✅ SOM ПЕРЕНАВЧЕНО УСПІШНО!")
    print("=" * 70)
    
    # Порівняння зі старою моделлю
    old_model_path = project_root / "models" / "som_optimized.pkl"
    if old_model_path.exists():
        print("\n📊 Порівняння зі старою моделлю:")
        print(f"   Стара: {old_model_path}")
        print(f"   Нова:  {output_model_path}")
        print("\n   Рекомендація: Протестуйте обидві моделі на реальних кейсах")


if __name__ == "__main__":
    main()
