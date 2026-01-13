"""Тест циклічної діагностики з вимкненим need_test"""

from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig

# Вимикаємо need_test для тесту
config = StoppingConfig(
    dominance_threshold=0.90,
    confidence_threshold=0.80,
    need_test_threshold=0.001,  # Майже вимкнено
    max_iterations=10
)

print("Завантаження моделей...")
controller = DiagnosisCycleController.from_models(
    database_path='data/unified_disease_symptom_merged.json',
    som_path='models/som_merged.pkl',
    nn_path='models/nn_two_branch.pt',
    stopping_config=config
)

print("\nСтарт сесії з симптомами: Fever, Cough, Headache")
result = controller.start_session(['Fever', 'Cough', 'Headache'])

print(f"\nStop reason: {result.stop_decision.reason.value if result.stop_decision else 'None'}")
print(f"Should continue: {controller.should_continue()}")

# Топ гіпотези
print("\nТоп-5 гіпотез:")
top5 = sorted(result.hypotheses.items(), key=lambda x: x[1], reverse=True)[:5]
for d, p in top5:
    print(f"  {d}: {p:.1%}")

# Пробуємо отримати питання
print("\n--- Цикл питань ---")
for i in range(5):
    if not controller.should_continue():
        print(f"Зупинка на ітерації {i}")
        break
    
    q = controller.get_next_question()
    if not q:
        print(f"Ітерація {i}: Питань немає")
        break
    
    print(f"\nQ{i+1}: {q.text}")
    print(f"  Симптом: {q.symptom}")
    print(f"  EIG: {q.eig:.4f}")
    
    # Відповідаємо "Так"
    r = controller.process_answer(True)
    
    top3 = sorted(r.hypotheses.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Після відповіді 'Так':")
    for d, p in top3:
        print(f"    {d}: {p:.1%}")

# Фінальний результат
print("\n--- Фінальний результат ---")
final = controller.get_result()
print(f"Stop reason: {final.stop_reason.value}")
print(f"Ітерацій: {final.iterations}")
print(f"Питань: {final.questions_asked}")
