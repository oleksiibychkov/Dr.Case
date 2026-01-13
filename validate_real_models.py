"""
Dr.Case — Валідація моделей (нова архітектура)

Перевіряє:
1. SOM якість (QE, TE, Fill)
2. NN якість (Accuracy, Top-5, Top-10)
3. Кластеризацію хвороб
4. Відповідність симптомів
"""

import json
import pickle
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict

print("=" * 60)
print("ВАЛІДАЦІЯ МОДЕЛЕЙ Dr.Case")
print("=" * 60)

# ============================================================
# 1. ЗАВАНТАЖЕННЯ
# ============================================================
print("\n[1] Завантаження...")

# База
with open('data/database_lowercase.json', 'r', encoding='utf-8') as f:
    db = json.load(f)
print(f"   База: {len(db)} хвороб")

# SOM
with open('models/som_real.pkl', 'rb') as f:
    som_data = pickle.load(f)

som = som_data['som']
unit_to_diseases = som_data['unit_to_diseases']
disease_to_unit = som_data['disease_to_unit']
disease_names = som_data['disease_names']

print(f"   SOM: {som.get_weights().shape[0]}x{som.get_weights().shape[1]}")
print(f"   Юнітів з хворобами: {len(unit_to_diseases)}")

# NN
checkpoint = torch.load('models/nn_real.pt', map_location='cpu', weights_only=False)
symptom_names = checkpoint['symptom_names']
print(f"   NN: {checkpoint['model_config']['n_symptoms']} симптомів → {checkpoint['model_config']['n_diseases']} хвороб")

# ============================================================
# 2. SOM ЯКІСТЬ
# ============================================================
print("\n[2] SOM якість...")

# Побудова векторів
symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}

def encode_disease_binary(disease_name: str) -> np.ndarray:
    """Бінарне кодування хвороби"""
    data = db[disease_name]
    symptoms = set(data.get('symptom_frequency', {}).keys())
    symptoms.update(data.get('symptoms', []))
    
    vector = np.zeros(len(symptom_names), dtype=np.float32)
    for symptom in symptoms:
        if symptom in symptom_to_idx:
            vector[symptom_to_idx[symptom]] = 1.0
    return vector

# Всі вектори
vectors = np.array([encode_disease_binary(d) for d in disease_names])

# Метрики
qe = som.quantization_error(vectors)
te = som.topographic_error(vectors)

# Fill ratio
filled = len(unit_to_diseases)
total = som.get_weights().shape[0] * som.get_weights().shape[1]
fill_ratio = filled / total

# Середня кількість хвороб на юніт
diseases_per_unit = [len(d) for d in unit_to_diseases.values()]
avg_diseases = np.mean(diseases_per_unit) if diseases_per_unit else 0

print(f"   QE: {qe:.4f} {'✅' if qe < 2.5 else '⚠️'}")
print(f"   TE: {te:.4f} {'✅' if te < 0.1 else '⚠️'}")
print(f"   Fill: {fill_ratio:.1%} {'✅' if 0.4 < fill_ratio < 0.8 else '⚠️'}")
print(f"   Хвороб/юніт: {avg_diseases:.1f}")

# ============================================================
# 3. КЛАСТЕРИЗАЦІЯ ПЕРЕВІРКА
# ============================================================
print("\n[3] Кластеризація...")

# Групи хвороб що мають бути поруч
disease_groups = {
    "респіраторні": ["covid-19", "influenza", "pneumonia", "bronchitis", "common cold"],
    "шкірні": ["acne", "eczema", "psoriasis", "dermatitis"],
    "серцево-судинні": ["hypertension", "heart failure", "myocardial infarction (heart attack)"],
    "неврологічні": ["migraine", "epilepsy", "parkinson's disease", "alzheimer's disease"],
}

print("   Перевірка груп хвороб:")
for group_name, diseases in disease_groups.items():
    existing = [d for d in diseases if d in disease_to_unit]
    if len(existing) < 2:
        print(f"   {group_name}: недостатньо хвороб в базі")
        continue
    
    # Отримуємо BMU для кожної хвороби
    bmus = [disease_to_unit[d] for d in existing]
    
    # Обчислюємо середню відстань між ними
    distances = []
    for i in range(len(bmus)):
        for j in range(i+1, len(bmus)):
            dist = abs(bmus[i][0] - bmus[j][0]) + abs(bmus[i][1] - bmus[j][1])
            distances.append(dist)
    
    avg_dist = np.mean(distances) if distances else 0
    status = "✅" if avg_dist < 6 else "⚠️"
    print(f"   {group_name}: avg_dist={avg_dist:.1f} {status}")
    for d in existing[:3]:
        print(f"      {d}: {disease_to_unit[d]}")

# ============================================================
# 4. NN ЯКІСТЬ (на згенерованих пацієнтах)
# ============================================================
print("\n[4] NN якість...")

from dr_case.neural_network.trainer_bmu import TwoBranchNN_BMU

# Завантаження моделі
model_config = checkpoint['model_config']
model = TwoBranchNN_BMU(
    n_symptoms=model_config['n_symptoms'],
    n_diseases=model_config['n_diseases'],
    symptom_hidden=model_config['symptom_hidden'],
    bmu_hidden=model_config['bmu_hidden'],
    combined_hidden=model_config['combined_hidden'],
    dropout=model_config['dropout']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Генерація тестових пацієнтів
n_test_per_disease = 10
correct_top1 = 0
correct_top5 = 0
correct_top10 = 0
total = 0

som_shape = (som.get_weights().shape[0], som.get_weights().shape[1])

print(f"   Тестування на {len(disease_names) * n_test_per_disease} пацієнтах...")

for i, disease_name in enumerate(disease_names):
    data = db[disease_name]
    freq_dict = data.get('symptom_frequency', {})
    if not freq_dict:
        freq_dict = {s: 1.0 for s in data.get('symptoms', [])}
    
    for _ in range(n_test_per_disease):
        # Генеруємо пацієнта
        patient_vector = np.zeros(len(symptom_names), dtype=np.float32)
        for symptom, freq in freq_dict.items():
            if symptom in symptom_to_idx:
                if freq > 1.0:
                    freq = freq / 100.0
                if np.random.random() < freq:
                    patient_vector[symptom_to_idx[symptom]] = 1.0
        
        # BMU
        bmu = som.winner(patient_vector)
        bmu_norm = np.array([bmu[0] / som_shape[0], bmu[1] / som_shape[1]], dtype=np.float32)
        
        # Prediction
        with torch.no_grad():
            logits = model(
                torch.FloatTensor(patient_vector).unsqueeze(0),
                torch.FloatTensor(bmu_norm).unsqueeze(0)
            )
            probs = torch.softmax(logits, dim=1)
            top_indices = probs.topk(10).indices[0].tolist()
        
        # Check
        if top_indices[0] == i:
            correct_top1 += 1
        if i in top_indices[:5]:
            correct_top5 += 1
        if i in top_indices[:10]:
            correct_top10 += 1
        total += 1

print(f"   Top-1 Accuracy: {correct_top1/total:.1%}")
print(f"   Top-5 Accuracy: {correct_top5/total:.1%}")
print(f"   Top-10 Accuracy: {correct_top10/total:.1%}")

# ============================================================
# 5. ТЕСТОВІ СЦЕНАРІЇ
# ============================================================
print("\n[5] Тестові сценарії...")

test_cases = [
    {
        "name": "Грип",
        "symptoms": ["fever", "cough", "muscle pain", "fatigue", "chills"],
        "expected": ["influenza", "covid-19"]
    },
    {
        "name": "COVID-19",
        "symptoms": ["fever", "cough", "loss of smell", "fatigue"],
        "expected": ["covid-19"]
    },
    {
        "name": "Застуда",
        "symptoms": ["runny nose", "sore throat", "sneezing"],
        "expected": ["common cold", "allergic rhinitis"]
    },
    {
        "name": "Головний біль",
        "symptoms": ["headache", "nausea", "sensitivity to light"],
        "expected": ["migraine"]
    },
]

for case in test_cases:
    # Кодуємо симптоми
    patient_vector = np.zeros(len(symptom_names), dtype=np.float32)
    for symptom in case["symptoms"]:
        if symptom in symptom_to_idx:
            patient_vector[symptom_to_idx[symptom]] = 1.0
    
    # BMU
    bmu = som.winner(patient_vector)
    bmu_norm = np.array([bmu[0] / som_shape[0], bmu[1] / som_shape[1]], dtype=np.float32)
    
    # Prediction
    with torch.no_grad():
        logits = model(
            torch.FloatTensor(patient_vector).unsqueeze(0),
            torch.FloatTensor(bmu_norm).unsqueeze(0)
        )
        probs = torch.softmax(logits, dim=1)
        top_indices = probs.topk(5).indices[0].tolist()
        top_probs = probs.topk(5).values[0].tolist()
    
    top_diseases = [(disease_names[idx], prob) for idx, prob in zip(top_indices, top_probs)]
    
    # Перевірка
    found = any(exp in [d[0] for d in top_diseases] for exp in case["expected"])
    status = "✅" if found else "❌"
    
    print(f"\n   {case['name']}: {status}")
    print(f"   Симптоми: {', '.join(case['symptoms'])}")
    print(f"   Очікувано: {case['expected']}")
    print(f"   Топ-3: {[(d, f'{p:.1%}') for d, p in top_diseases[:3]]}")

# ============================================================
# ПІДСУМОК
# ============================================================
print("\n" + "=" * 60)
print("ПІДСУМОК ВАЛІДАЦІЇ")
print("=" * 60)
print(f"SOM: QE={qe:.4f}, TE={te:.4f}, Fill={fill_ratio:.1%}")
print(f"NN:  Top-1={correct_top1/total:.1%}, Top-5={correct_top5/total:.1%}, Top-10={correct_top10/total:.1%}")
print("=" * 60)
