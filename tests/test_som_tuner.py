"""
Тести для модуля optimization/som_tuner

Запуск: pytest tests/test_som_tuner.py -v
Або демо: python tests/test_som_tuner.py
"""

from pathlib import Path
import tempfile

# Шлях до бази даних
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"


def test_tuner_creation():
    """Тест створення SOMTuner"""
    from dr_case.optimization import SOMTuner
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    tuner = SOMTuner(
        database_path=str(DATA_PATH),
        target_recall=0.99,
        max_candidates_ratio=0.15
    )
    
    assert tuner.n_diseases == 842
    assert tuner.n_symptoms == 461
    
    print(f"✓ SOMTuner created: {tuner}")
    print(f"  Train size: {len(tuner.train_indices)}")
    print(f"  Validation size: {len(tuner.val_indices)}")
    
    return tuner


def test_quick_optimization():
    """Тест швидкої оптимізації (мало trials)"""
    from dr_case.optimization import SOMTuner
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    tuner = SOMTuner(
        database_path=str(DATA_PATH),
        target_recall=0.95,  # Нижчий поріг для швидкого тесту
        max_candidates_ratio=0.20
    )
    
    # Запускаємо з мінімумом trials
    print("\n--- Running quick optimization (5 trials) ---")
    result = tuner.quick_tune(n_trials=5)
    
    assert result.best_value > 0
    assert "recall" in result.best_metrics
    assert result.n_trials == 5
    
    print(f"\n✓ Quick optimization complete")
    print(f"  Best score: {result.best_value:.4f}")
    print(f"  Best recall: {result.best_metrics['recall']:.2%}")
    print(f"  Best candidates: {result.best_metrics['avg_candidates']:.1f}")
    print(f"  Best grid: {result.best_params['grid_size']}x{result.best_params['grid_size']}")
    
    return result


def test_get_best_model():
    """Тест отримання найкращої моделі"""
    from dr_case.optimization import SOMTuner
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    tuner = SOMTuner(database_path=str(DATA_PATH))
    
    # Швидка оптимізація
    tuner.quick_tune(n_trials=3)
    
    # Отримуємо модель
    som, projector = tuner.get_best_model()
    
    assert som.is_trained
    assert projector is not None
    
    print(f"✓ Got best model: {som}")
    
    return som, projector


def test_train_with_best_params():
    """Тест навчання з найкращими параметрами"""
    from dr_case.optimization import SOMTuner
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    tuner = SOMTuner(database_path=str(DATA_PATH))
    
    # Швидка оптимізація
    result = tuner.quick_tune(n_trials=3)
    
    # Навчаємо фінальну модель
    print("\n--- Training final model ---")
    som, projector, metrics = tuner.train_with_best_params()
    
    assert som.is_trained
    assert metrics["recall"] > 0
    
    print(f"✓ Final model trained")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  Avg candidates: {metrics['avg_candidates']:.1f}")
    
    return som, projector, metrics


def test_save_load_result():
    """Тест збереження/завантаження результату"""
    from dr_case.optimization import SOMTuner, TuningResult
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    tuner = SOMTuner(database_path=str(DATA_PATH))
    result = tuner.quick_tune(n_trials=3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tuning_result.json"
        
        # Зберігаємо
        result.save(str(path))
        assert path.exists()
        
        # Завантажуємо
        loaded = TuningResult.load(str(path))
        
        assert loaded.best_value == result.best_value
        assert loaded.best_params == result.best_params
        
        print(f"✓ Save/Load test passed")
    
    return True


def demo():
    """Демонстрація модуля som_tuner"""
    print("=" * 60)
    print("Dr.Case — Демонстрація SOM Tuner")
    print("=" * 60)
    
    # Перевірка наявності бази даних
    if not DATA_PATH.exists():
        print(f"\n❌ ПОМИЛКА: База даних не знайдена!")
        print(f"   Очікуваний шлях: {DATA_PATH}")
        return False
    
    try:
        # Перевірка Optuna
        try:
            import optuna
            print(f"\n✓ Optuna version: {optuna.__version__}")
        except ImportError:
            print("\n❌ Optuna не встановлено. Виконайте: pip install optuna")
            return False
        
        print("\n--- 1. Creating SOMTuner ---")
        tuner = test_tuner_creation()
        
        if tuner is None:
            return False
        
        print("\n--- 2. Quick Optimization ---")
        result = test_quick_optimization()
        
        print("\n--- 3. Train Final Model ---")
        test_train_with_best_params()
        
        print("\n--- 4. Save/Load Result ---")
        test_save_load_result()
        
        print("\n" + "=" * 60)
        print("✅ Всі тести пройдено успішно!")
        print("=" * 60)
        
        # Показуємо рекомендації
        if result:
            print("\n📋 Рекомендовані параметри:")
            print(f"   Grid size: {result.best_params['grid_size']}x{result.best_params['grid_size']}")
            print(f"   Epochs: {result.best_params['epochs']}")
            print(f"   Alpha: {result.best_params['alpha']:.3f}")
            print(f"   K: {result.best_params['k']}")
            print(f"   Tau: {result.best_params['tau']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_optimization(n_trials: int = 30):
    """
    Повна оптимізація (запускати окремо, займає ~10-30 хвилин)
    
    Використання:
        python -c "from tests.test_som_tuner import full_optimization; full_optimization(50)"
    """
    from dr_case.optimization import SOMTuner
    
    print("=" * 60)
    print("Dr.Case — Повна оптимізація SOM")
    print("=" * 60)
    
    tuner = SOMTuner(
        database_path=str(DATA_PATH),
        target_recall=0.995,
        max_candidates_ratio=0.12
    )
    
    # Оптимізація
    result = tuner.optimize(n_trials=n_trials)
    
    # Зберігаємо результат
    result.save("models/som_tuning_result.json")
    print(f"\n✓ Result saved to models/som_tuning_result.json")
    
    # Навчаємо фінальну модель
    som, projector, metrics = tuner.train_with_best_params()
    
    # Зберігаємо модель
    som.save("models/som_optimized.pkl")
    print(f"✓ Model saved to models/som_optimized.pkl")
    
    return result


if __name__ == "__main__":
    demo()
