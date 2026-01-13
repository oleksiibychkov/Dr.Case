"""Debug тест - що насправді робить cycle_controller"""
import numpy as np
from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig

config = StoppingConfig(
    dominance_threshold=0.99,
    need_test_threshold=0.001,
    max_iterations=2
)

print("Завантаження...")
controller = DiagnosisCycleController.from_models(
    database_path='data/unified_disease_symptom_merged.json',
    som_path='models/som_merged.pkl',
    nn_path='models/nn_two_branch.pt',
    stopping_config=config
)

# Перевіряємо чи symptom_names завантажено
print(f"\n=== DEBUG ===")
print(f"nn_model має symptom_names: {hasattr(controller.nn_model, 'symptom_names')}")
if hasattr(controller.nn_model, 'symptom_names'):
    names = controller.nn_model.symptom_names
    print(f"  Кількість: {len(names) if names else 0}")
    print(f"  Перші 5: {names[:5] if names else 'N/A'}")
    print(f"  'fever' в списку: {'fever' in names if names else False}")
else:
    print("  symptom_names НЕ ЗАВАНТАЖЕНО!")

# Старт сесії
print(f"\n=== Старт сесії ===")
result = controller.start_session(['Fever', 'Cough', 'Headache'])

# Дивимось session_state
print(f"\nknown_symptoms: {controller.session_state.known_symptoms}")

# Дивимось що передається в NN
print(f"\n=== Перевірка вектора ===")
# Симулюємо те що робить _predict_hypotheses
if hasattr(controller.nn_model, 'symptom_names') and controller.nn_model.symptom_names:
    nn_symptom_names = controller.nn_model.symptom_names
    nn_symptom_to_idx = {s: i for i, s in enumerate(nn_symptom_names)}
    nn_vector = np.zeros(len(nn_symptom_names), dtype=np.float32)
    
    print(f"Конвертуємо симптоми:")
    for symptom in controller.session_state.known_symptoms:
        key = symptom.lower()
        found = key in nn_symptom_to_idx
        print(f"  '{symptom}' -> '{key}': {'знайдено' if found else 'НЕ знайдено'}")
        if found:
            nn_vector[nn_symptom_to_idx[key]] = 1.0
    
    print(f"\nАктивні індекси: {np.where(nn_vector == 1)[0].tolist()}")
else:
    print("symptom_names не доступні - використовується fallback!")

# Результати
print(f"\n=== Результати ===")
top5 = sorted(result.hypotheses.items(), key=lambda x: x[1], reverse=True)[:5]
for d, p in top5:
    print(f"  {d}: {p:.2%}")
