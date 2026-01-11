"""
Dr.Case — Тест оптимізації Neural Network (Етап 11)

Запуск:
    python test_nn_tuner.py

Цей скрипт:
1. Завантажує базу даних діагнозів
2. Запускає Optuna для пошуку оптимальних гіперпараметрів NN
3. Навчає фінальну модель з найкращими параметрами
4. Зберігає результат у models/nn_tuning_result.json
"""

import sys
from pathlib import Path

# Додаємо шлях до пакету dr_case
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 60)
    print("Dr.Case — Neural Network Optimization (Stage 11)")
    print("=" * 60)
    
    # Імпортуємо після додавання шляху
    from dr_case.optimization import NNTuner
    
    # Шлях до бази даних
    database_path = project_root / "data" / "unified_disease_symptom_data_full.json"
    
    if not database_path.exists():
        # Альтернативний шлях для Windows
        database_path = Path(r"C:\Projects\Dr.Case\data\unified_disease_symptom_data_full.json")
    
    if not database_path.exists():
        print(f"ERROR: Database not found at {database_path}")
        print("Please update the path to your database file.")
        return
    
    print(f"\nDatabase: {database_path}")
    
    # Створюємо тюнер
    tuner = NNTuner(
        database_path=str(database_path),
        target_recall_at_5=0.95,
        target_recall_at_1=0.70,
        validation_split=0.2,
        max_epochs_per_trial=30,  # Менше епох для швидшого пошуку
        random_seed=42
    )
    
    print(f"\n{tuner}")
    
    # === Швидка оптимізація (для тесту) ===
    print("\n--- Quick Tune (20 trials) ---")
    result = tuner.quick_tune(n_trials=20)
    
    # Зберігаємо результат
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    
    result.save(output_dir / "nn_tuning_result.json")
    
    # === Навчаємо фінальну модель ===
    print("\n--- Training Final Model ---")
    final_model, final_metrics = tuner.train_with_best_params(epochs=100)
    
    # Зберігаємо модель
    tuner.save_best_model(output_dir / "nn_model.pt")
    
    # === Підсумок ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best parameters saved: {output_dir / 'nn_tuning_result.json'}")
    print(f"Model saved: {output_dir / 'nn_model.pt'}")
    print(f"\nFinal Metrics:")
    print(f"  Recall@1:  {final_metrics['recall_at_1']:.2%}")
    print(f"  Recall@5:  {final_metrics['recall_at_5']:.2%}")
    print(f"  Recall@10: {final_metrics['recall_at_10']:.2%}")
    print("=" * 60)


def test_quick():
    """Швидкий тест без повного навчання"""
    print("=" * 60)
    print("Dr.Case — Quick Test (5 trials)")
    print("=" * 60)
    
    from dr_case.optimization import NNTuner
    
    database_path = Path(r"C:\Projects\Dr.Case\data\unified_disease_symptom_data_full.json")
    
    if not database_path.exists():
        print(f"ERROR: Database not found at {database_path}")
        return
    
    tuner = NNTuner(
        database_path=str(database_path),
        max_epochs_per_trial=10,  # Дуже мало для тесту
        random_seed=42
    )
    
    # Тільки 5 trials для швидкого тесту
    result = tuner.optimize(n_trials=5, study_name="nn_quick_test")
    
    print(f"\nQuick test completed!")
    print(f"Best Recall@5: {result.best_metrics['recall_at_5']:.2%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        test_quick()
    else:
        main()
