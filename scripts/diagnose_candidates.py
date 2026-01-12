#!/usr/bin/env python3
"""
Діагностика Candidate Selection
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 60)
    print("ДІАГНОСТИКА CANDIDATE SELECTION")
    print("=" * 60)
    
    # Завантажуємо дані
    som_path = project_root / "models" / "som_merged.pkl"
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    with open(som_path, 'rb') as f:
        som_data = pickle.load(f)
    
    with open(db_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    som = som_data['som']
    unit_to_diseases = som_data['unit_to_diseases']
    disease_to_unit = som_data.get('disease_to_unit', {})
    
    h, w = som._weights.shape[:2]
    
    print(f"\n📊 SOM: {h}x{w} = {h*w} юнітів")
    print(f"📊 Активних юнітів: {len(unit_to_diseases)}")
    print(f"📊 Хвороб в базі: {len(database)}")
    
    # Аналіз ключів
    sample_keys = list(unit_to_diseases.keys())[:5]
    print(f"\n📊 Приклади ключів unit_to_diseases:")
    for k in sample_keys:
        print(f"   {k} (type: {type(k).__name__})")
    
    # Тест на 20 випадках
    print(f"\n" + "=" * 60)
    print("ТЕСТ CANDIDATE SELECTION (20 випадків)")
    print("=" * 60)
    
    np.random.seed(42)
    diseases = list(database.keys())
    test_diseases = np.random.choice(diseases, size=20, replace=False)
    
    # Словник симптомів
    all_symptoms = set()
    for d in database.values():
        all_symptoms.update(d.get('symptoms', []))
    symptom_list = sorted([s.lower() for s in all_symptoms])
    symptom_to_idx = {s: i for i, s in enumerate(symptom_list)}
    n_symptoms = len(symptom_list)
    
    hits = 0
    misses = 0
    
    for disease in test_diseases:
        symptoms = database[disease].get('symptoms', [])
        if not symptoms:
            continue
        
        # Кодуємо симптоми
        vec = np.zeros(n_symptoms, dtype=np.float32)
        for s in symptoms[:5]:
            s_lower = s.strip().lower()
            if s_lower in symptom_to_idx:
                vec[symptom_to_idx[s_lower]] = 1.0
        
        # Нормалізуємо
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        # Знаходимо BMU та топ-k юнітів
        distances = []
        for i in range(h):
            for j in range(w):
                weight = som._weights[i, j]
                dist = np.linalg.norm(vec - weight)
                distances.append(((i, j), dist))
        
        distances.sort(key=lambda x: x[1])
        
        # Топ-10 юнітів
        top_k = 10
        active_units = [d[0] for d in distances[:top_k]]
        
        # Збираємо кандидатів
        candidates = set()
        for unit_tuple in active_units:
            i, j = unit_tuple
            
            # Пробуємо різні формати
            found = False
            
            # Варіант 1: tuple
            if unit_tuple in unit_to_diseases:
                candidates.update(unit_to_diseases[unit_tuple])
                found = True
            
            # Варіант 2: tuple з int
            if not found and (int(i), int(j)) in unit_to_diseases:
                candidates.update(unit_to_diseases[(int(i), int(j))])
                found = True
            
            # Варіант 3: пошук серед ключів
            if not found:
                for key in unit_to_diseases.keys():
                    if isinstance(key, tuple) and len(key) == 2:
                        if int(key[0]) == i and int(key[1]) == j:
                            candidates.update(unit_to_diseases[key])
                            found = True
                            break
        
        # Перевіряємо чи є правильний діагноз
        if disease in candidates:
            hits += 1
            status = "✅"
        else:
            misses += 1
            status = "❌"
            
            # Де насправді цей діагноз?
            correct_unit = disease_to_unit.get(disease)
            print(f"\n{status} {disease}")
            print(f"   Симптоми: {symptoms[:3]}...")
            print(f"   Кандидатів: {len(candidates)}")
            print(f"   Топ-3 юніти: {active_units[:3]}")
            print(f"   Правильний юніт: {correct_unit}")
            
            if correct_unit:
                # Де цей юніт в рейтингу?
                for rank, (unit, dist) in enumerate(distances):
                    if isinstance(correct_unit, tuple):
                        if int(unit[0]) == int(correct_unit[0]) and int(unit[1]) == int(correct_unit[1]):
                            print(f"   Позиція правильного юніта: {rank + 1}")
                            break
                    else:
                        flat = unit[0] * w + unit[1]
                        if flat == correct_unit:
                            print(f"   Позиція правильного юніта: {rank + 1}")
                            break
    
    recall = hits / (hits + misses)
    print(f"\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТ")
    print(f"=" * 60)
    print(f"   Hits: {hits}/{hits + misses}")
    print(f"   Recall: {recall:.1%}")
    
    if recall < 0.9:
        print(f"\n⚠️ Recall низький!")
        print(f"   Можливі рішення:")
        print(f"   1. Збільшити top_k (зараз 10)")
        print(f"   2. Перенавчити SOM з кращими параметрами")
        print(f"   3. Додати cumulative mass selection")


if __name__ == "__main__":
    main()
