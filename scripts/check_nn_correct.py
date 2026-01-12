#!/usr/bin/env python3
"""
Правильна перевірка Neural Network з коректним форматом даних
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dr_case.neural_network import TwoBranchNN


def generate_som_context(som, symptom_vector: np.ndarray, som_dim: int = 10) -> np.ndarray:
    """Генерація SOM context як при навчанні"""
    # Обчислюємо відстані до всіх юнітів
    h, w = som._weights.shape[:2]
    distances = []
    
    for i in range(h):
        for j in range(w):
            w_vec = som._weights[i, j]
            dist = np.linalg.norm(symptom_vector - w_vec)
            distances.append(dist)
    
    # Сортуємо
    sorted_indices = np.argsort(distances)
    top_k_dists = np.array([distances[i] for i in sorted_indices[:som_dim]])
    
    # Softmax membership (як при навчанні)
    temperature = 1.0
    exp_neg = np.exp(-top_k_dists / temperature)
    membership = exp_neg / exp_neg.sum()
    
    return membership.astype(np.float32)


def main():
    print("=" * 60)
    print("ПРАВИЛЬНА ПЕРЕВІРКА NEURAL NETWORK")
    print("=" * 60)
    
    # Завантажуємо моделі
    nn_path = project_root / "models" / "nn_two_branch.pt"
    som_path = project_root / "models" / "som_merged.pkl"
    db_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
    
    with open(som_path, 'rb') as f:
        som_data = pickle.load(f)
    som = som_data['som']
    
    with open(db_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    # Конфігурація
    model_config = checkpoint.get('model_config', {})
    n_symptoms = model_config.get('n_symptoms', 460)
    n_diseases = model_config.get('n_diseases', 844)
    som_dim = model_config.get('som_dim', 10)
    
    print(f"\n📊 Конфігурація:")
    print(f"   n_symptoms: {n_symptoms}")
    print(f"   n_diseases: {n_diseases}")
    print(f"   som_dim: {som_dim}")
    
    # Завантажуємо модель
    model = TwoBranchNN(
        n_symptoms=n_symptoms,
        som_dim=som_dim,
        n_diseases=n_diseases
    )
    
    state_dict = checkpoint.get('model_state') or checkpoint.get('model_state_dict')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Словник симптомів (як при навчанні - lowercase)
    symptom_names = checkpoint.get('symptom_names', [])
    disease_names = checkpoint.get('disease_names', [])
    
    print(f"\n📊 symptom_names (перші 5): {symptom_names[:5]}")
    print(f"📊 disease_names (перші 5): {disease_names[:5]}")
    
    # Тестовий випадок
    test_disease = list(database.keys())[0]
    test_symptoms_raw = database[test_disease].get('symptoms', [])[:5]
    
    print(f"\n🧪 Тестовий випадок:")
    print(f"   Хвороба: {test_disease}")
    print(f"   Симптоми (raw): {test_symptoms_raw}")
    
    # Кодуємо симптоми (lowercase, як при навчанні)
    symptom_vector = np.zeros(n_symptoms, dtype=np.float32)
    found_count = 0
    
    for symptom in test_symptoms_raw:
        symptom_lower = symptom.strip().lower()
        if symptom_lower in symptom_names:
            idx = symptom_names.index(symptom_lower)
            symptom_vector[idx] = 1.0
            found_count += 1
    
    print(f"   Знайдено симптомів: {found_count}/{len(test_symptoms_raw)}")
    
    # L2 нормалізація (як при навчанні)
    norm = np.linalg.norm(symptom_vector)
    if norm > 0:
        symptom_vector_norm = symptom_vector / norm
    else:
        symptom_vector_norm = symptom_vector
    
    print(f"   L2 norm: {norm:.4f}")
    print(f"   Після нормалізації: sum={symptom_vector_norm.sum():.4f}, norm={np.linalg.norm(symptom_vector_norm):.4f}")
    
    # SOM context (як при навчанні)
    som_context = generate_som_context(som, symptom_vector_norm, som_dim)
    
    print(f"\n📊 SOM context:")
    print(f"   Values: {som_context}")
    print(f"   Sum: {som_context.sum():.4f}")
    
    # Передбачення
    x_symptoms = torch.tensor(symptom_vector_norm, dtype=torch.float32).unsqueeze(0)
    x_som = torch.tensor(som_context, dtype=torch.float32).unsqueeze(0)
    
    print(f"\n📊 Вхідні тензори:")
    print(f"   symptoms shape: {x_symptoms.shape}, norm: {torch.norm(x_symptoms).item():.4f}")
    print(f"   som shape: {x_som.shape}, sum: {x_som.sum().item():.4f}")
    
    with torch.no_grad():
        logits = model(x_symptoms, x_som)
        
        print(f"\n📊 Logits:")
        print(f"   mean: {logits.mean().item():.4f}")
        print(f"   std: {logits.std().item():.4f}")
        print(f"   range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        # SOFTMAX (не sigmoid!) - бо CrossEntropyLoss при навчанні
        probs = torch.softmax(logits, dim=-1)
        
        print(f"\n📊 Probabilities (softmax):")
        print(f"   sum: {probs.sum().item():.4f}")  # Має бути ≈1.0
        print(f"   max: {probs.max().item():.6f}")
        print(f"   min: {probs.min().item():.10f}")
        
        # Топ-10
        top_vals, top_idx = torch.topk(probs.squeeze(), 10)
        
        print(f"\n📊 Топ-10 передбачень:")
        for i, (idx, val) in enumerate(zip(top_idx.numpy(), top_vals.numpy())):
            name = disease_names[idx] if idx < len(disease_names) else f"Disease_{idx}"
            marker = "✅" if name == test_disease else ""
            print(f"   {i+1}. {name}: {val:.6f} {marker}")
        
        # Позиція правильної відповіді
        if test_disease in disease_names:
            correct_idx = disease_names.index(test_disease)
            correct_prob = probs[0, correct_idx].item()
            rank = (probs[0] > correct_prob).sum().item() + 1
            print(f"\n📊 Правильна відповідь '{test_disease}':")
            print(f"   Позиція: {rank}")
            print(f"   Ймовірність: {correct_prob:.6f}")
        else:
            print(f"\n⚠️ '{test_disease}' не знайдено в disease_names!")
    
    # Тест на кількох випадках
    print("\n" + "=" * 60)
    print("📊 ТЕСТ НА 20 ВИПАДКАХ")
    print("=" * 60)
    
    np.random.seed(42)
    test_diseases = np.random.choice(list(database.keys()), size=20, replace=False)
    
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    
    for disease in test_diseases:
        symptoms = database[disease].get('symptoms', [])
        if not symptoms:
            continue
        
        # Кодуємо
        vec = np.zeros(n_symptoms, dtype=np.float32)
        for s in symptoms[:5]:
            s_lower = s.strip().lower()
            if s_lower in symptom_names:
                vec[symptom_names.index(s_lower)] = 1.0
        
        # Нормалізуємо
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        
        # SOM context
        som_ctx = generate_som_context(som, vec, som_dim)
        
        # Передбачення
        x_s = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        x_som = torch.tensor(som_ctx, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(x_s, x_som)
            probs = torch.softmax(logits, dim=-1)
            
            top_idx = torch.topk(probs.squeeze(), 10).indices.numpy()
            top_names = [disease_names[i] for i in top_idx]
            
            if disease in top_names[:1]:
                top1_hits += 1
            if disease in top_names[:5]:
                top5_hits += 1
            if disease in top_names[:10]:
                top10_hits += 1
    
    print(f"\n📊 Результати:")
    print(f"   Accuracy@1:  {top1_hits}/20 = {top1_hits/20*100:.1f}%")
    print(f"   Accuracy@5:  {top5_hits}/20 = {top5_hits/20*100:.1f}%")
    print(f"   Accuracy@10: {top10_hits}/20 = {top10_hits/20*100:.1f}%")
    
    if top5_hits >= 15:
        print("\n✅ Модель працює коректно!")
    else:
        print("\n⚠️ Точність нижча за очікувану")


if __name__ == "__main__":
    main()
