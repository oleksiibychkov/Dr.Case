"""Порівняння результатів з SOM membership і без"""
import torch
import numpy as np
from dr_case.neural_network.two_branch_model import TwoBranchNN

# Завантажуємо модель
checkpoint = torch.load('models/nn_two_branch.pt', map_location='cpu')
config = checkpoint['model_config']
symptom_names = checkpoint['symptom_names']
disease_names = checkpoint['disease_names']
symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}

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

# Симптоми
symptoms = ['fever', 'cough', 'headache', 'fatigue']
x = np.zeros(config['n_symptoms'], dtype=np.float32)
for s in symptoms:
    x[symptom_to_idx[s]] = 1.0

x_tensor = torch.tensor(x).unsqueeze(0)

print(f"Симптоми: {symptoms}")
print(f"Активні індекси: {np.where(x == 1)[0].tolist()}")

# Тест 1: SOM = zeros
print(f"\n=== SOM = zeros ===")
som_zero = torch.zeros(1, config['som_dim'])
with torch.no_grad():
    probs = torch.softmax(model(x_tensor, som_zero), dim=-1).squeeze().numpy()
results = sorted([(disease_names[i], probs[i]) for i in range(len(probs))], key=lambda x: -x[1])[:5]
for d, p in results:
    print(f"  {d}: {p:.2%}")

# Тест 2: SOM = реалістичний membership (як в cycle_controller)
print(f"\n=== SOM = realistic membership ===")
# Типовий membership: перший юніт домінує
som_real = torch.tensor([[0.4, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.0]])
with torch.no_grad():
    probs = torch.softmax(model(x_tensor, som_real), dim=-1).squeeze().numpy()
results = sorted([(disease_names[i], probs[i]) for i in range(len(probs))], key=lambda x: -x[1])[:5]
for d, p in results:
    print(f"  {d}: {p:.2%}")

# Тест 3: SOM = рівномірний
print(f"\n=== SOM = uniform (0.1 each) ===")
som_uniform = torch.ones(1, config['som_dim']) * 0.1
with torch.no_grad():
    probs = torch.softmax(model(x_tensor, som_uniform), dim=-1).squeeze().numpy()
results = sorted([(disease_names[i], probs[i]) for i in range(len(probs))], key=lambda x: -x[1])[:5]
for d, p in results:
    print(f"  {d}: {p:.2%}")

# Тест 4: SOM = ones
print(f"\n=== SOM = ones ===")
som_ones = torch.ones(1, config['som_dim'])
with torch.no_grad():
    probs = torch.softmax(model(x_tensor, som_ones), dim=-1).squeeze().numpy()
results = sorted([(disease_names[i], probs[i]) for i in range(len(probs))], key=lambda x: -x[1])[:5]
for d, p in results:
    print(f"  {d}: {p:.2%}")
