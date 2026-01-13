"""Тест NN з lowercase симптомами"""
import torch
import numpy as np

# Завантажуємо NN
checkpoint = torch.load('models/nn_two_branch.pt', map_location='cpu')
config = checkpoint['model_config']
symptom_names = checkpoint['symptom_names']
disease_names = checkpoint['disease_names']

symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}

print(f"Симптомів: {len(symptom_names)}")
print(f"Хвороб: {len(disease_names)}")
print(f"\nПерші 5 симптомів: {symptom_names[:5]}")

# Перевіряємо чи є потрібні симптоми (lowercase)
test_symptoms = ['fever', 'cough', 'headache']
print(f"\nШукаємо симптоми:")
for sym in test_symptoms:
    found = sym in symptom_to_idx
    print(f"  '{sym}': {'знайдено' if found else 'НЕ знайдено'}")

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

# Кодуємо симптоми (lowercase!)
x = np.zeros(config['n_symptoms'], dtype=np.float32)
for sym in test_symptoms:
    if sym in symptom_to_idx:
        x[symptom_to_idx[sym]] = 1.0

print(f"\nАктивні індекси: {np.where(x == 1)[0].tolist()}")

# Предикт
x_tensor = torch.tensor(x).unsqueeze(0)
som_tensor = torch.zeros(1, config['som_dim'])

with torch.no_grad():
    outputs = model(x_tensor, som_tensor)
    probs = torch.softmax(outputs, dim=-1).squeeze().numpy()

# Топ-10
results = [(disease_names[i], probs[i]) for i in range(len(probs))]
results.sort(key=lambda x: x[1], reverse=True)

print(f"\n=== Топ-10 для {test_symptoms} ===")
for disease, prob in results[:10]:
    print(f"  {disease}: {prob:.2%}")

# Тепер додаємо fatigue
print(f"\n=== Додаємо 'fatigue' ===")
if 'fatigue' in symptom_to_idx:
    x[symptom_to_idx['fatigue']] = 1.0
    print(f"  'fatigue' знайдено, idx={symptom_to_idx['fatigue']}")
else:
    print(f"  'fatigue' НЕ знайдено!")

print(f"Активні індекси: {np.where(x == 1)[0].tolist()}")

x_tensor = torch.tensor(x).unsqueeze(0)
with torch.no_grad():
    outputs = model(x_tensor, som_tensor)
    probs = torch.softmax(outputs, dim=-1).squeeze().numpy()

results = [(disease_names[i], probs[i]) for i in range(len(probs))]
results.sort(key=lambda x: x[1], reverse=True)

print(f"\n=== Топ-10 для {test_symptoms + ['fatigue']} ===")
for disease, prob in results[:10]:
    print(f"  {disease}: {prob:.2%}")
