"""Тест Neural Network окремо"""
import json
import torch
import numpy as np

# Завантажуємо базу
with open('data/unified_disease_symptom_merged.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

# Симптоми з бази
all_symptoms = set()
for disease, info in db.items():
    if isinstance(info, dict) and 'symptoms' in info:
        all_symptoms.update(info['symptoms'])
symptom_list = sorted(all_symptoms)
symptom_to_idx = {s: i for i, s in enumerate(symptom_list)}

print(f"Симптомів: {len(symptom_list)}")
print(f"Хвороб: {len(db)}")

# Завантажуємо NN
checkpoint = torch.load('models/nn_two_branch.pt', map_location='cpu')
config = checkpoint['model_config']
disease_names = checkpoint['disease_names']
symptom_names = checkpoint['symptom_names']

print(f"\nNN config:")
print(f"  n_symptoms: {config['n_symptoms']}")
print(f"  n_diseases: {config['n_diseases']}")
print(f"  disease_names в checkpoint: {len(disease_names)}")
print(f"  symptom_names в checkpoint: {len(symptom_names)}")

# Порівнюємо симптоми
print(f"\nСимптоми з checkpoint vs з бази:")
checkpoint_syms = set(symptom_names)
db_syms = set(symptom_list)
print(f"  В checkpoint: {len(checkpoint_syms)}")
print(f"  В базі: {len(db_syms)}")
print(f"  Спільних: {len(checkpoint_syms & db_syms)}")

# Кодуємо тестові симптоми (використовуємо symptom_names з checkpoint!)
test_symptoms = ['Fever', 'Cough', 'Headache']
checkpoint_sym_to_idx = {s: i for i, s in enumerate(symptom_names)}

x = np.zeros(config['n_symptoms'], dtype=np.float32)
for sym in test_symptoms:
    if sym in checkpoint_sym_to_idx:
        x[checkpoint_sym_to_idx[sym]] = 1.0
        print(f"  {sym}: idx={checkpoint_sym_to_idx[sym]}")
    else:
        print(f"  {sym}: NOT FOUND in checkpoint!")

# Будуємо модель
from dr_case.neural_network.two_branch_model import TwoBranchNN

model = TwoBranchNN(
    n_symptoms=config['n_symptoms'],
    n_diseases=config['n_diseases'],
    som_dim=config['som_dim'],
    symptom_hidden=config['symptom_hidden'],
    som_hidden=config['som_hidden'],
    combined_hidden=config['combined_hidden']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Предикт
x_tensor = torch.tensor(x).unsqueeze(0)
som_tensor = torch.zeros(1, config['som_dim'])  # Нульовий SOM membership

with torch.no_grad():
    probs = model(x_tensor, som_tensor).squeeze().numpy()

# Топ-10
idx_to_disease = {i: d for i, d in enumerate(disease_names)}
results = [(idx_to_disease[i], probs[i]) for i in range(len(probs))]
results.sort(key=lambda x: x[1], reverse=True)

print(f"\n=== Топ-10 для {test_symptoms} ===")
for disease, prob in results[:10]:
    print(f"  {disease}: {prob:.2%}")

# Тепер додаємо Fatigue
print(f"\n=== Додаємо Fatigue ===")
if 'Fatigue' in checkpoint_sym_to_idx:
    x[checkpoint_sym_to_idx['Fatigue']] = 1.0
    print(f"  Fatigue: idx={checkpoint_sym_to_idx['Fatigue']}")

x_tensor = torch.tensor(x).unsqueeze(0)
with torch.no_grad():
    probs = model(x_tensor, som_tensor).squeeze().numpy()

results = [(idx_to_disease[i], probs[i]) for i in range(len(probs))]
results.sort(key=lambda x: x[1], reverse=True)

print(f"\n=== Топ-10 для {test_symptoms + ['Fatigue']} ===")
for disease, prob in results[:10]:
    print(f"  {disease}: {prob:.2%}")
