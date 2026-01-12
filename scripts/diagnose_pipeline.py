#!/usr/bin/env python3
"""
Dr.Case — ДІАГНОСТИКА PIPELINE
==============================

Крок за кроком перевіряємо кожен компонент:
1. База даних симптомів
2. SOM unit_to_diseases
3. Candidate selection
4. NN predictions

Запуск:
    python scripts/diagnose_pipeline.py
"""

import sys
from pathlib import Path
import json
import pickle
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f"🔍 {text}")
    print("=" * 70)


def diagnose_database():
    """Перевірка бази даних"""
    print_header("КРОК 1: База даних")
    
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if not db_path.exists():
        print(f"   ❌ Файл не знайдено: {db_path}")
        return None
    
    with open(db_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    print(f"   ✅ Завантажено: {len(database)} хвороб")
    
    # Статистика симптомів
    all_symptoms = set()
    symptoms_per_disease = []
    
    for disease, data in database.items():
        symptoms = data.get('symptoms', [])
        all_symptoms.update(symptoms)
        symptoms_per_disease.append(len(symptoms))
    
    print(f"   📊 Унікальних симптомів: {len(all_symptoms)}")
    print(f"   📊 Симптомів на хворобу: min={min(symptoms_per_disease)}, max={max(symptoms_per_disease)}, avg={np.mean(symptoms_per_disease):.1f}")
    
    # Приклад хвороби
    example = list(database.keys())[0]
    print(f"\n   📋 Приклад: {example}")
    print(f"      Симптоми: {database[example].get('symptoms', [])[:5]}...")
    
    return database


def diagnose_som():
    """Перевірка SOM"""
    print_header("КРОК 2: SOM Model")
    
    som_path = project_root / "models" / "som_merged.pkl"
    
    if not som_path.exists():
        print(f"   ❌ Файл не знайдено: {som_path}")
        return None
    
    with open(som_path, 'rb') as f:
        som_data = pickle.load(f)
    
    print(f"   ✅ Ключі checkpoint: {list(som_data.keys())}")
    
    # unit_to_diseases
    unit_to_diseases = som_data.get('unit_to_diseases', {})
    print(f"   📊 unit_to_diseases: {len(unit_to_diseases)} юнітів")
    
    if unit_to_diseases:
        # Скільки хвороб в юнітах
        diseases_counts = [len(d) for d in unit_to_diseases.values()]
        total_diseases = sum(diseases_counts)
        print(f"   📊 Всього хвороб у юнітах: {total_diseases}")
        print(f"   📊 Хвороб на юніт: min={min(diseases_counts)}, max={max(diseases_counts)}, avg={np.mean(diseases_counts):.1f}")
        
        # Приклад юніта
        example_unit = list(unit_to_diseases.keys())[0]
        example_diseases = unit_to_diseases[example_unit]
        print(f"\n   📋 Приклад юніт {example_unit}: {example_diseases[:3]}...")
    
    # disease_names
    disease_names = som_data.get('disease_names', [])
    print(f"\n   📊 disease_names: {len(disease_names)} хвороб")
    if disease_names:
        print(f"      Перші 5: {disease_names[:5]}")
    
    # SOM object
    som = som_data.get('som')
    if som is not None:
        print(f"\n   📊 SOM grid: {som._weights.shape}")
    
    return som_data


def diagnose_nn():
    """Перевірка NN"""
    print_header("КРОК 3: Neural Network")
    
    nn_path = project_root / "models" / "nn_two_branch.pt"
    
    if not nn_path.exists():
        print(f"   ❌ Файл не знайдено: {nn_path}")
        return None
    
    import torch
    checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
    
    print(f"   ✅ Ключі checkpoint: {list(checkpoint.keys())}")
    
    model_config = checkpoint.get('model_config', {})
    print(f"\n   📊 model_config:")
    for k, v in model_config.items():
        print(f"      {k}: {v}")
    
    disease_names = checkpoint.get('disease_names', [])
    symptom_names = checkpoint.get('symptom_names', [])
    
    print(f"\n   📊 disease_names: {len(disease_names)}")
    print(f"   📊 symptom_names: {len(symptom_names)}")
    
    if disease_names:
        print(f"      Перші 5 diseases: {disease_names[:5]}")
    if symptom_names:
        print(f"      Перші 5 symptoms: {symptom_names[:5]}")
    
    return checkpoint


def diagnose_candidate_selection(database, som_data):
    """Перевірка candidate selection"""
    print_header("КРОК 4: Candidate Selection (ручна перевірка)")
    
    if not database or not som_data:
        print("   ❌ Потрібні database та som_data")
        return
    
    from dr_case.encoding import SymptomVocabulary
    
    # Словник симптомів
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    vocab = SymptomVocabulary.from_database(str(db_path))
    print(f"   📊 Vocabulary: {vocab.size} симптомів")
    
    # Беремо тестову хворобу
    test_disease = list(database.keys())[0]
    test_symptoms = database[test_disease].get('symptoms', [])[:5]
    
    print(f"\n   🧪 Тест: {test_disease}")
    print(f"      Симптоми: {test_symptoms}")
    
    # Кодуємо симптоми
    symptom_vector = np.zeros(vocab.size)
    found_symptoms = []
    missing_symptoms = []
    
    for symptom in test_symptoms:
        if vocab.has_symptom(symptom):
            idx = vocab.symptom_to_index(symptom)
            symptom_vector[idx] = 1.0
            found_symptoms.append(symptom)
        else:
            missing_symptoms.append(symptom)
    
    print(f"\n   📊 Знайдено у словнику: {len(found_symptoms)}/{len(test_symptoms)}")
    if missing_symptoms:
        print(f"   ⚠️ Не знайдено: {missing_symptoms}")
    
    # Проєкція на SOM
    som = som_data.get('som')
    if som is None:
        print("   ❌ SOM object не знайдено")
        return
    
    # Нормалізація
    norm = np.linalg.norm(symptom_vector)
    if norm > 0:
        symptom_vector_norm = symptom_vector / norm
    else:
        symptom_vector_norm = symptom_vector
    
    # BMU
    bmu = som.winner(symptom_vector_norm)
    print(f"\n   📊 BMU (Best Matching Unit): {bmu}")
    
    # Конвертуємо в індекс
    grid_h, grid_w = som._weights.shape[:2]
    bmu_idx = bmu[0] * grid_w + bmu[1]
    print(f"   📊 BMU index: {bmu_idx}")
    
    # Перевіряємо unit_to_diseases
    unit_to_diseases = som_data.get('unit_to_diseases', {})
    
    # Ключі можуть бути різними
    if bmu_idx in unit_to_diseases:
        candidates = unit_to_diseases[bmu_idx]
    elif str(bmu_idx) in unit_to_diseases:
        candidates = unit_to_diseases[str(bmu_idx)]
    elif bmu in unit_to_diseases:
        candidates = unit_to_diseases[bmu]
    elif str(bmu) in unit_to_diseases:
        candidates = unit_to_diseases[str(bmu)]
    else:
        candidates = []
        print(f"\n   ⚠️ Юніт {bmu_idx} не знайдено в unit_to_diseases!")
        print(f"      Типи ключів: {type(list(unit_to_diseases.keys())[0]) if unit_to_diseases else 'empty'}")
        print(f"      Приклад ключа: {list(unit_to_diseases.keys())[:3]}")
    
    print(f"\n   📊 Кандидати з юніта {bmu_idx}: {len(candidates)}")
    if candidates:
        print(f"      Перші 5: {candidates[:5]}")
        
        # Чи є цільова хвороба серед кандидатів?
        if test_disease in candidates:
            print(f"\n   ✅ {test_disease} ЗНАЙДЕНО серед кандидатів!")
        else:
            print(f"\n   ❌ {test_disease} НЕ знайдено серед кандидатів!")
            
            # Де має бути ця хвороба?
            disease_to_unit = som_data.get('disease_to_unit', {})
            if test_disease in disease_to_unit:
                correct_unit = disease_to_unit[test_disease]
                print(f"      Правильний юніт: {correct_unit}")
            elif disease_names := som_data.get('disease_names', []):
                if test_disease in disease_names:
                    print(f"      Хвороба є в disease_names")


def diagnose_nn_prediction(database, som_data, nn_checkpoint):
    """Перевірка NN передбачень"""
    print_header("КРОК 5: NN Prediction (ручна перевірка)")
    
    if not all([database, som_data, nn_checkpoint]):
        print("   ❌ Потрібні всі компоненти")
        return
    
    import torch
    from dr_case.neural_network import TwoBranchNN
    from dr_case.encoding import SymptomVocabulary
    
    # Завантажуємо модель
    model_config = nn_checkpoint.get('model_config', {})
    n_symptoms = model_config.get('n_symptoms', 460)
    n_diseases = model_config.get('n_diseases', 844)
    som_dim = model_config.get('som_dim', 10)
    
    model = TwoBranchNN(
        n_symptoms=n_symptoms,
        som_dim=som_dim,
        n_diseases=n_diseases
    )
    
    state_dict = nn_checkpoint.get('model_state') or nn_checkpoint.get('model_state_dict')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"   ✅ Модель завантажена: {n_symptoms} symptoms, {n_diseases} diseases")
    
    # Тестовий вектор
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    vocab = SymptomVocabulary.from_database(str(db_path))
    
    test_disease = list(database.keys())[0]
    test_symptoms = database[test_disease].get('symptoms', [])[:5]
    
    print(f"\n   🧪 Тест: {test_disease}")
    print(f"      Симптоми: {test_symptoms}")
    
    # Кодуємо симптоми
    symptom_vector = np.zeros(n_symptoms)
    for symptom in test_symptoms:
        if vocab.has_symptom(symptom):
            idx = vocab.symptom_to_index(symptom)
            if idx < n_symptoms:
                symptom_vector[idx] = 1.0
    
    # SOM membership (спрощено)
    som_membership = np.zeros(som_dim)
    som_membership[0] = 1.0  # BMU
    
    # Передбачення
    with torch.no_grad():
        x_symptoms = torch.FloatTensor(symptom_vector).unsqueeze(0)
        x_som = torch.FloatTensor(som_membership).unsqueeze(0)
        
        output = model(x_symptoms, x_som)
        probs = torch.sigmoid(output).squeeze().numpy()
    
    # Топ-10 передбачень
    disease_names = nn_checkpoint.get('disease_names', [])
    
    top_indices = np.argsort(probs)[-10:][::-1]
    
    print(f"\n   📊 Топ-10 передбачень:")
    for i, idx in enumerate(top_indices):
        if idx < len(disease_names):
            name = disease_names[idx]
            prob = probs[idx]
            marker = "✅" if name == test_disease else ""
            print(f"      {i+1}. {name}: {prob:.4f} {marker}")
    
    # Позиція правильної відповіді
    if test_disease in disease_names:
        correct_idx = disease_names.index(test_disease)
        correct_prob = probs[correct_idx]
        rank = (probs > correct_prob).sum() + 1
        print(f"\n   📊 Правильна відповідь '{test_disease}':")
        print(f"      Позиція: {rank}")
        print(f"      Ймовірність: {correct_prob:.4f}")
    else:
        print(f"\n   ❌ '{test_disease}' не знайдено в disease_names!")


def main():
    print("=" * 70)
    print("Dr.Case — ДІАГНОСТИКА PIPELINE")
    print("=" * 70)
    
    # Крок 1
    database = diagnose_database()
    
    # Крок 2
    som_data = diagnose_som()
    
    # Крок 3
    nn_checkpoint = diagnose_nn()
    
    # Крок 4
    if database and som_data:
        diagnose_candidate_selection(database, som_data)
    
    # Крок 5
    if database and som_data and nn_checkpoint:
        diagnose_nn_prediction(database, som_data, nn_checkpoint)
    
    print_header("ВИСНОВКИ")
    print("""
   Можливі причини Recall = 0:
   
   1. ❓ Симптоми не збігаються між базою та словником
   2. ❓ unit_to_diseases має неправильний формат ключів
   3. ❓ disease_names в SOM ≠ disease_names в NN
   4. ❓ NN не навчена (всі ймовірності ~однакові)
   5. ❓ Тестові дані генеруються неправильно
   
   Запусти цей скрипт і перевір кожен крок!
""")


if __name__ == "__main__":
    main()
