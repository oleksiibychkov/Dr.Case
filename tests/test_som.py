"""
Тести для модуля som

Запуск: pytest tests/test_som.py -v
Або демо: python tests/test_som.py
"""

import numpy as np
from pathlib import Path
import tempfile

# Шлях до бази даних
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"


def test_som_model_creation():
    """Тест створення SOMModel"""
    from dr_case.config import SOMConfig
    from dr_case.som import SOMModel
    
    config = SOMConfig(
        grid_height=10,
        grid_width=10,
        input_dim=50
    )
    
    som = SOMModel(config)
    
    assert not som.is_trained
    assert som.grid_shape == (10, 10)
    
    print(f"✓ SOMModel created: {som}")
    
    return True


def test_som_training():
    """Тест навчання SOM на синтетичних даних"""
    from dr_case.config import SOMConfig
    from dr_case.som import SOMModel
    
    # Синтетичні дані
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    data = np.random.rand(n_samples, n_features).astype(np.float32)
    disease_names = [f"disease_{i}" for i in range(n_samples)]
    
    # Конфігурація
    config = SOMConfig(
        grid_height=10,
        grid_width=10,
        input_dim=n_features,
        epochs=100
    )
    
    # Навчання
    som = SOMModel(config)
    metrics = som.train(data, disease_names, verbose=False)
    
    assert som.is_trained
    assert "qe" in metrics
    assert "te" in metrics
    assert "fill_ratio" in metrics
    
    print(f"✓ SOM trained on synthetic data")
    print(f"  QE: {metrics['qe']:.4f}")
    print(f"  TE: {metrics['te']:.4f}")
    print(f"  Fill ratio: {metrics['fill_ratio']:.2%}")
    
    return som, data, disease_names


def test_som_projection():
    """Тест проєкції на SOM"""
    from dr_case.config import CandidateSelectorConfig
    from dr_case.som import SOMProjector
    
    # Отримуємо навчену модель
    som, data, disease_names = test_som_training()
    
    # Створюємо проєктор
    selector_config = CandidateSelectorConfig(alpha=0.9, k=6, tau=0.01)
    projector = SOMProjector(som, selector_config)
    
    # Проєкція першого вектора
    result = projector.project(data[0])
    
    assert result.bmu is not None
    assert len(result.bmu) == 2
    assert result.bmu_distance >= 0
    assert len(result.memberships) > 0
    assert len(result.active_units) > 0
    assert disease_names[0] in result.candidate_diseases
    
    print(f"✓ Projection test passed")
    print(f"  BMU: {result.bmu}")
    print(f"  BMU distance: {result.bmu_distance:.4f}")
    print(f"  Active units: {len(result.active_units)}")
    print(f"  Candidates: {len(result.candidate_diseases)}")
    print(f"  Cumulative mass: {result.cumulative_mass:.4f}")
    
    return True


def test_som_save_load():
    """Тест збереження/завантаження"""
    from dr_case.som import SOMModel
    
    # Навчаємо
    som, data, disease_names = test_som_training()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "som_model.pkl"
        
        # Зберігаємо
        som.save(str(path))
        assert path.exists()
        
        # Завантажуємо
        loaded_som = SOMModel.load(str(path))
        
        assert loaded_som.is_trained
        assert loaded_som.grid_shape == som.grid_shape
        assert loaded_som.quantization_error == som.quantization_error
        
        # Перевіряємо що проєкція працює
        bmu_original = som.get_bmu(data[0])
        bmu_loaded = loaded_som.get_bmu(data[0])
        
        assert bmu_original == bmu_loaded
        
        print(f"✓ Save/Load test passed")
    
    return True


def test_trainer_with_real_data():
    """Тест SOMTrainer з реальними даними"""
    from dr_case.som import SOMTrainer
    from dr_case.config import SOMConfig
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping real data test: {DATA_PATH} not found")
        return None
    
    # Налаштування для швидкого тесту
    config = SOMConfig(
        grid_height=15,
        grid_width=15,
        epochs=100  # Менше епох для швидкості
    )
    
    trainer = SOMTrainer(config)
    
    # Навчання
    print("\n--- Training SOM on real data ---")
    som, metrics = trainer.train_from_database(str(DATA_PATH), verbose=True)
    
    assert som.is_trained
    assert metrics['qe'] > 0
    
    print(f"✓ SOM trained on real data")
    print(f"  Diseases: {len(trainer.disease_names)}")
    print(f"  Grid: {config.grid_height}x{config.grid_width}")
    
    return trainer


def test_candidate_recall():
    """Тест Candidate Recall"""
    trainer = test_trainer_with_real_data()
    
    if trainer is None:
        return None
    
    print("\n--- Evaluating Candidate Recall ---")
    
    # Тестуємо з різними параметрами
    params_to_test = [
        {"alpha": 0.9, "k": 6, "tau": 0.01},
        {"alpha": 0.95, "k": 8, "tau": 0.01},
        {"alpha": 0.99, "k": 10, "tau": 0.005},
    ]
    
    for params in params_to_test:
        recall_metrics = trainer.evaluate_candidate_recall(**params)
        
        print(f"\nα={params['alpha']}, k={params['k']}, τ={params['tau']}:")
        print(f"  Recall: {recall_metrics['recall']:.2%}")
        print(f"  Avg candidates: {recall_metrics['avg_candidates']:.1f}")
    
    return True


def test_patient_projection():
    """Тест проєкції симптомів пацієнта"""
    trainer = test_trainer_with_real_data()
    
    if trainer is None:
        return None
    
    print("\n--- Testing Patient Projection ---")
    
    from dr_case.encoding import PatientEncoder
    
    # Створюємо енкодер пацієнта
    patient_encoder = PatientEncoder(trainer.encoder.vocabulary)
    
    # Симулюємо пацієнта з симптомами грипу
    patient_symptoms = ["fever", "headache", "cough", "fatigue"]
    patient_vector = patient_encoder.encode(patient_symptoms)
    
    # Проєктуємо
    projector = trainer.get_projector(alpha=0.9, k=6, tau=0.01)
    result = projector.project(patient_vector)
    
    print(f"Patient symptoms: {patient_symptoms}")
    print(f"BMU: {result.bmu}")
    print(f"Active units: {len(result.active_units)}")
    print(f"Candidates: {len(result.candidate_diseases)}")
    print(f"Top 5 candidates:")
    for i, disease in enumerate(result.candidate_diseases[:5]):
        print(f"  {i+1}. {disease}")
    
    # Конвертуємо в SOMResult
    som_result = result.to_som_result()
    print(f"\n✓ SOMResult: BMU={som_result.bmu_id}, active_units={som_result.active_units_count}")
    
    return True


def demo():
    """Повна демонстрація модуля som"""
    print("=" * 60)
    print("Dr.Case — Демонстрація модуля SOM")
    print("=" * 60)
    
    try:
        print("\n--- 1. SOMModel Creation ---")
        test_som_model_creation()
        
        print("\n--- 2. SOM Training (synthetic) ---")
        test_som_training()
        
        print("\n--- 3. SOM Projection ---")
        test_som_projection()
        
        print("\n--- 4. Save/Load ---")
        test_som_save_load()
        
        print("\n--- 5. Real Data Training ---")
        trainer = test_trainer_with_real_data()
        
        if trainer:
            print("\n--- 6. Candidate Recall ---")
            test_candidate_recall()
            
            print("\n--- 7. Patient Projection ---")
            test_patient_projection()
        
        print("\n" + "=" * 60)
        print("✅ Всі тести пройдено успішно!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo()
