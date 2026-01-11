"""
Тести для модуля neural_network

Запуск: pytest tests/test_neural_network.py -v
Або демо: python tests/test_neural_network.py
"""

import tempfile
from pathlib import Path
import numpy as np

# Шляхи
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "nn_model.pt"


def test_model_creation():
    """Тест створення моделі"""
    from dr_case.neural_network import DiagnosisNN
    
    model = DiagnosisNN(
        n_symptoms=461,
        n_diseases=842,
        hidden_dims=[256, 128]
    )
    
    print(f"✓ Model created: {model.count_parameters():,} parameters")
    print(f"  Architecture: {model.hidden_dims}")
    
    # Тест forward pass
    import torch
    x = torch.randn(4, 461)
    out = model(x)
    
    assert out.shape == (4, 842)
    print(f"✓ Forward pass: input {x.shape} → output {out.shape}")
    
    return model


def test_model_with_mask():
    """Тест моделі з маскою кандидатів"""
    from dr_case.neural_network import DiagnosisNN
    import torch
    
    model = DiagnosisNN(n_symptoms=461, n_diseases=842)
    
    x = torch.randn(2, 461)
    
    # Маска: тільки діагнози 0, 5, 10 є кандидатами
    mask = torch.zeros(2, 842)
    mask[:, [0, 5, 10]] = 1
    
    out = model(x, mask)
    
    # Перевіряємо що не-кандидати мають -inf
    assert torch.isinf(out[:, 1]).all()  # не кандидат
    assert not torch.isinf(out[:, 0]).any()  # кандидат
    
    print(f"✓ Mask test passed")
    print(f"  Candidate scores: {out[0, [0, 5, 10]].tolist()}")
    
    return True


def test_trainer_creation():
    """Тест створення trainer"""
    from dr_case.neural_network import NNTrainer, TrainingConfig
    
    config = TrainingConfig(
        hidden_dims=[256, 128],
        epochs=5,
        batch_size=32
    )
    
    trainer = NNTrainer(
        n_symptoms=461,
        n_diseases=842,
        config=config
    )
    
    print(f"✓ Trainer created: {trainer}")
    
    return trainer


def test_quick_training():
    """Тест швидкого навчання на реальних даних"""
    from dr_case.neural_network import NNTrainer, TrainingConfig
    from dr_case.encoding import DiseaseEncoder
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    # Завантажуємо дані
    encoder = DiseaseEncoder.from_database(str(DATA_PATH))
    disease_matrix = encoder.encode_all()
    
    print(f"Data: {disease_matrix.shape}")
    
    # Конфігурація для швидкого тесту
    config = TrainingConfig(
        hidden_dims=[256, 128],
        epochs=20,  # Було 3, тепер 20
        batch_size=64,
        patience=10,  # Додаємо patience до 10
        augment=True
    )
    
    trainer = NNTrainer(
        n_symptoms=encoder.vector_dim,
        n_diseases=encoder.disease_count,
        config=config
    )
    
    # Навчання
    print("\n--- Training (3 epochs) ---")
    history = trainer.train(
        disease_matrix,
        encoder.disease_names,
        val_split=0.2,
        verbose=True
    )
    
    assert len(history['train_loss']) > 0
    assert history['val_acc'][-1] > 0
    
    print(f"\n✓ Training complete")
    print(f"  Final val accuracy: {history['val_acc'][-1]:.2%}")
    
    return trainer, encoder


def test_save_load():
    """Тест збереження/завантаження"""
    from dr_case.neural_network import NNTrainer, TrainingConfig
    from dr_case.encoding import DiseaseEncoder
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    # Швидке навчання
    encoder = DiseaseEncoder.from_database(str(DATA_PATH))
    disease_matrix = encoder.encode_all()
    
    config = TrainingConfig(hidden_dims=[128], epochs=10) # Було 2, тепер 10
    trainer = NNTrainer(
        n_symptoms=encoder.vector_dim,
        n_diseases=encoder.disease_count,
        config=config
    )
    trainer.train(disease_matrix, encoder.disease_names, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt"
        
        # Зберігаємо
        trainer.save(str(path))
        assert path.exists()
        
        # Завантажуємо
        loaded = NNTrainer.load(str(path))
        
        assert loaded.n_symptoms == trainer.n_symptoms
        assert loaded.n_diseases == trainer.n_diseases
        assert len(loaded.disease_names) == len(trainer.disease_names)
        
        print(f"✓ Save/Load test passed")
    
    return True


def test_ranker():
    """Тест DiagnosisRanker"""
    from dr_case.neural_network import NNTrainer, DiagnosisRanker, TrainingConfig
    from dr_case.encoding import DiseaseEncoder, SymptomVocabulary
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    # Завантажуємо дані
    encoder = DiseaseEncoder.from_database(str(DATA_PATH))
    disease_matrix = encoder.encode_all()
    vocabulary = SymptomVocabulary.from_database(str(DATA_PATH))
    
    # Швидке навчання
    config = TrainingConfig(hidden_dims=[128], epochs=10)  # Було 2, тепер 10
    trainer = NNTrainer(
        n_symptoms=encoder.vector_dim,
        n_diseases=encoder.disease_count,
        config=config
    )
    trainer.train(disease_matrix, encoder.disease_names, verbose=False)
    
    # Створюємо Ranker
    ranker = DiagnosisRanker(trainer, vocabulary)
    
    print(f"✓ DiagnosisRanker created: {ranker}")
    
    # Тест ранжування
    symptoms = ["fever", "headache", "cough"]
    candidates = encoder.disease_names[:20]  # Перші 20 діагнозів як кандидати
    
    result = ranker.rank(symptoms, candidates)
    
    assert len(result.ranked_diseases) > 0
    assert len(result.scores) == len(result.ranked_diseases)
    
    print(f"\n✓ Ranking test")
    print(f"  Symptoms: {symptoms}")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Top 5:")
    for disease, score, prob in result.get_top_n(5):
        print(f"    {disease}: {prob:.2%}")
    
    return ranker


def test_ranker_all():
    """Тест ранжування всіх діагнозів"""
    from dr_case.neural_network import NNTrainer, DiagnosisRanker, TrainingConfig
    from dr_case.encoding import DiseaseEncoder, SymptomVocabulary
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    # Завантажуємо та навчаємо
    encoder = DiseaseEncoder.from_database(str(DATA_PATH))
    disease_matrix = encoder.encode_all()
    vocabulary = SymptomVocabulary.from_database(str(DATA_PATH))
    
    config = TrainingConfig(hidden_dims=[128], epochs=10) # Було 2, тепер 10
    trainer = NNTrainer(
        n_symptoms=encoder.vector_dim,
        n_diseases=encoder.disease_count,
        config=config
    )
    trainer.train(disease_matrix, encoder.disease_names, verbose=False)
    
    ranker = DiagnosisRanker(trainer, vocabulary)
    
    # Ранжуємо ВСІ діагнози
    symptoms = ["fever", "cough", "fatigue"]
    result = ranker.rank_all(symptoms, top_n=10)
    
    assert len(result.ranked_diseases) == 10
    
    print(f"\n✓ Rank all test")
    print(f"  Symptoms: {symptoms}")
    print(f"  Top 10 from ALL {encoder.disease_count} diseases:")
    for i, (disease, score, prob) in enumerate(result.get_top_n(10)):
        print(f"    {i+1}. {disease}: {prob:.2%}")
    
    return result


def demo():
    """Повна демонстрація модуля neural_network"""
    print("=" * 60)
    print("Dr.Case — Демонстрація Neural Network")
    print("=" * 60)
    
    try:
        # Перевірка PyTorch
        import torch
        print(f"\n✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        print("\n--- 1. Model Creation ---")
        test_model_creation()
        
        print("\n--- 2. Model with Mask ---")
        test_model_with_mask()
        
        print("\n--- 3. Trainer Creation ---")
        test_trainer_creation()
        
        if DATA_PATH.exists():
            print("\n--- 4. Quick Training ---")
            test_quick_training()
            
            print("\n--- 5. Save/Load ---")
            test_save_load()
            
            print("\n--- 6. DiagnosisRanker ---")
            test_ranker()
            
            print("\n--- 7. Rank All Diseases ---")
            test_ranker_all()
        else:
            print(f"\n⚠ Skipping data tests: {DATA_PATH} not found")
        
        print("\n" + "=" * 60)
        print("✅ Всі тести пройдено успішно!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_training(epochs: int = 50):
    """
    Повне навчання моделі (запускати окремо).
    
    Використання:
        python -c "from tests.test_neural_network import full_training; full_training(50)"
    """
    from dr_case.neural_network import NNTrainer, TrainingConfig
    from dr_case.encoding import DiseaseEncoder
    
    print("=" * 60)
    print("Dr.Case — Повне навчання Neural Network")
    print("=" * 60)
    
    # Завантажуємо дані
    encoder = DiseaseEncoder.from_database(str(DATA_PATH))
    disease_matrix = encoder.encode_all()
    
    print(f"Data: {disease_matrix.shape}")
    
    # Конфігурація
    config = TrainingConfig(
        hidden_dims=[512, 256, 128],
        epochs=epochs,
        batch_size=64,
        learning_rate=1e-3,
        dropout=0.3,
        patience=15,
        augment=True,
        augment_prob=0.5,
        use_scheduler=True
    )
    
    trainer = NNTrainer(
        n_symptoms=encoder.vector_dim,
        n_diseases=encoder.disease_count,
        config=config
    )
    
    # Навчання
    history = trainer.train(
        disease_matrix,
        encoder.disease_names,
        val_split=0.2,
        verbose=True
    )
    
    # Зберігаємо
    Path("models").mkdir(exist_ok=True)
    trainer.save("models/nn_model.pt")
    
    print(f"\n✓ Model saved to models/nn_model.pt")
    print(f"  Final val accuracy: {history['val_acc'][-1]:.2%}")
    
    return trainer, history


if __name__ == "__main__":
    demo()
