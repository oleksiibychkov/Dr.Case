"""Тест конвертації регістру для нових симптомів"""
from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig
import numpy as np

# Конфіг з вимкненим need_test
config = StoppingConfig(
    dominance_threshold=0.90,
    confidence_threshold=0.80,
    need_test_threshold=0.001,  # Майже вимкнено
    max_iterations=10
)

print("Завантаження...")
controller = DiagnosisCycleController.from_models(
    database_path='data/unified_disease_symptom_merged.json',
    som_path='models/som_merged.pkl',
    nn_path='models/nn_two_branch.pt',
    stopping_config=config
)

result = controller.start_session(['Fever', 'Cough', 'Headache'])
print(f'\nПочаток: {list(controller.session_state.known_symptoms)}')

# Топ-3 до відповіді
top3 = sorted(result.hypotheses.items(), key=lambda x: x[1], reverse=True)[:3]
print(f'Топ-3: {[(d, f"{p:.1%}") for d, p in top3]}')

# Питання
q = controller.get_next_question()
if q is None:
    print(f'\nПитання: None')
    print(f'stop_decision: {result.stop_decision}')
    exit()

print(f'\nПитання: {q.text}')
print(f'Симптом: {q.symptom}')

# Відповідь Так
result2 = controller.process_answer(True)
print(f'\nПісля відповіді "Так":')
print(f'known_symptoms: {list(controller.session_state.known_symptoms)}')

# Перевіряємо конвертацію
nn_syms = controller.nn_model.symptom_names
nn_idx = {s: i for i, s in enumerate(nn_syms)}

print(f'\nКонвертація в lowercase:')
for s in controller.session_state.known_symptoms:
    low = s.lower()
    found = low in nn_idx
    print(f'  "{s}" -> "{low}": {"OK" if found else "NOT FOUND"}')

# Топ-3 після
top3 = sorted(result2.hypotheses.items(), key=lambda x: x[1], reverse=True)[:3]
print(f'\nТоп-3 після: {[(d, f"{p:.1%}") for d, p in top3]}')

# Перевіримо вектор напряму
print(f'\n=== Перевірка вектора для NN ===')
nn_vector = np.zeros(len(nn_syms), dtype=np.float32)
for s in controller.session_state.known_symptoms:
    key = s.lower()
    if key in nn_idx:
        nn_vector[nn_idx[key]] = 1.0

active = np.where(nn_vector == 1)[0].tolist()
print(f'Активні індекси: {active}')
print(f'Симптоми: {[nn_syms[i] for i in active]}')
