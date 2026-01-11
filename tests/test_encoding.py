"""
Тести для модуля encoding

Запуск: pytest tests/test_encoding.py -v
Або демо: python tests/test_encoding.py
"""

import numpy as np
from pathlib import Path

# Шлях до бази даних
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"


def test_data_loader():
    """Тест завантаження бази даних"""
    from dr_case.encoding import DiseaseDatabaseLoader
    
    loader = DiseaseDatabaseLoader(str(DATA_PATH))
    
    print(f"✓ Завантажено діагнозів: {loader.disease_count}")
    print(f"✓ Унікальних симптомів: {loader.symptom_count}")
    
    assert loader.disease_count > 0, "База порожня!"
    assert loader.symptom_count > 0, "Немає симптомів!"
    
    # Статистика
    stats = loader.get_statistics()
    print(f"✓ Середня к-ть симптомів на діагноз: {stats['avg_symptoms_per_disease']:.1f}")
    print(f"✓ Мін симптомів: {stats['min_symptoms']}, Макс: {stats['max_symptoms']}")
    
    return loader


def test_symptom_vocabulary(loader):
    """Тест словника симптомів"""
    from dr_case.encoding import SymptomVocabulary
    
    vocab = SymptomVocabulary.from_database(str(DATA_PATH))
    
    print(f"✓ Розмір словника: {vocab.size}")
    
    assert vocab.size == loader.symptom_count
    
    # Перевірка індексації
    first_symptom = vocab.index_to_symptom(0)
    print(f"✓ Перший симптом (idx=0): {first_symptom}")
    
    idx = vocab.symptom_to_index(first_symptom)
    assert idx == 0, "Індексація порушена!"
    
    # Перевірка пошуку
    symptoms = vocab.symptoms[:5]
    print(f"✓ Перші 5 симптомів: {symptoms}")
    
    return vocab


def test_disease_encoder(loader, vocab):
    """Тест кодування діагнозів"""
    from dr_case.encoding import DiseaseEncoder
    
    encoder = DiseaseEncoder(vocab, loader)
    
    print(f"✓ DiseaseEncoder: {encoder}")
    
    # Візьмемо перший діагноз
    disease_name = loader.disease_names[0]
    print(f"✓ Тестовий діагноз: {disease_name}")
    
    # Кодуємо
    vector = encoder.encode(disease_name)
    assert vector is not None, "Не вдалося закодувати діагноз"
    assert vector.shape == (vocab.size,), f"Неправильна форма: {vector.shape}"
    
    # Перевіряємо що вектор не нульовий
    nonzero = np.sum(vector > 0)
    print(f"✓ Вектор має {nonzero} ненульових елементів (симптомів)")
    
    # Декодуємо назад
    decoded = encoder.decode(vector)
    original = loader.get_symptoms(disease_name)
    print(f"✓ Оригінальних симптомів: {len(original)}, декодовано: {len(decoded)}")
    
    # Матриця всіх діагнозів
    matrix = encoder.encode_all()
    print(f"✓ Матриця всіх діагнозів: shape {matrix.shape}")
    assert matrix.shape == (loader.disease_count, vocab.size)
    
    return encoder


def test_patient_encoder(vocab):
    """Тест кодування пацієнта"""
    from dr_case.encoding import PatientEncoder
    
    encoder = PatientEncoder(vocab)
    print(f"✓ PatientEncoder: {encoder}")
    
    # Візьмемо декілька симптомів зі словника
    test_symptoms = vocab.symptoms[:3]
    print(f"✓ Тестові симптоми: {test_symptoms}")
    
    # Бінарне кодування
    vector = encoder.encode(test_symptoms)
    assert vector.shape == (vocab.size,)
    assert np.sum(vector > 0) == len(test_symptoms)
    print(f"✓ Бінарне кодування: {np.sum(vector > 0)} активних")
    
    # Вагове кодування
    weighted = {s: 0.5 + 0.1*i for i, s in enumerate(test_symptoms)}
    vector_weighted = encoder.encode_weighted(weighted)
    print(f"✓ Вагове кодування: {np.sum(vector_weighted > 0)} активних")
    
    # З негативними симптомами
    present = test_symptoms[:2]
    absent = test_symptoms[2:3]
    vector_neg = encoder.encode_with_negatives(present, absent)
    print(f"✓ З негативними: присутніх={np.sum(vector_neg > 0)}, відсутніх={np.sum(vector_neg < 0)}")
    
    return encoder


def demo():
    """Повна демонстрація модуля encoding"""
    print("=" * 60)
    print("Dr.Case — Демонстрація модуля encoding")
    print("=" * 60)
    
    # Перевірка наявності бази даних
    if not DATA_PATH.exists():
        print(f"\n❌ ПОМИЛКА: База даних не знайдена!")
        print(f"   Очікуваний шлях: {DATA_PATH}")
        print(f"   Покладіть файл unified_disease_symptom_data_full.json в папку data/")
        return False
    
    print(f"\n📂 База даних: {DATA_PATH}")
    
    try:
        # 1. Завантаження
        print("\n--- 1. Завантаження бази даних ---")
        loader = test_data_loader()
        
        # 2. Словник
        print("\n--- 2. Словник симптомів ---")
        vocab = test_symptom_vocabulary(loader)
        
        # 3. Disease Encoder
        print("\n--- 3. Кодування діагнозів ---")
        disease_encoder = test_disease_encoder(loader, vocab)
        
        # 4. Patient Encoder
        print("\n--- 4. Кодування пацієнта ---")
        patient_encoder = test_patient_encoder(vocab)
        
        # 5. Приклад використання
        print("\n--- 5. Приклад: пошук схожих діагнозів ---")
        
        # Беремо симптоми пацієнта
        patient_symptoms = vocab.symptoms[:5]  # Перші 5 симптомів
        patient_vector = patient_encoder.encode(patient_symptoms)
        
        # Порівнюємо з усіма діагнозами
        disease_matrix = disease_encoder.encode_all()
        
        # Cosine similarity
        patient_norm = patient_vector / (np.linalg.norm(patient_vector) + 1e-8)
        disease_norms = disease_matrix / (np.linalg.norm(disease_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = disease_norms @ patient_norm
        
        # Топ-5 діагнозів
        top_indices = np.argsort(similarities)[::-1][:5]
        print(f"Симптоми пацієнта: {patient_symptoms}")
        print("Топ-5 найбільш схожих діагнозів:")
        for i, idx in enumerate(top_indices):
            disease_name = disease_encoder.index_to_disease(idx)
            print(f"  {i+1}. {disease_name}: {similarities[idx]:.3f}")
        
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
