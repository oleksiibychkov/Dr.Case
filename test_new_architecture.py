"""
Тест нової архітектури з BMU координатами

Перевіряє:
1. Завантаження нової NN моделі (nn_lowercase.pt)
2. BMU координати
3. Розділення гіпотез на "з кластера" / "не з кластера"
4. Трекінг діагнозів не з кластера
"""

from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig

print("=" * 60)
print("Тест нової архітектури з BMU координатами")
print("=" * 60)

# Конфіг
config = StoppingConfig(
    dominance_threshold=0.85,
    need_test_threshold=0.05,
    max_iterations=10
)

# Завантаження — НОВІ LOWERCASE МОДЕЛІ
print("\n[1] Завантаження моделей...")
controller = DiagnosisCycleController.from_models(
    database_path='data/database_lowercase.json',
    som_path='models/som_real.pkl',
    nn_path='models/nn_real.pt',
    stopping_config=config
)

# Старт сесії — LOWERCASE симптоми!
print("\n[2] Старт сесії з симптомами: fever, cough, headache")
result = controller.start_session(['fever', 'cough', 'headache'])

print(f"\n=== Ітерація {result.iteration} ===")
print(f"BMU координати: {result.bmu_coords}")
print(f"BMU нормалізовані: ({result.bmu_coords_normalized[0]:.3f}, {result.bmu_coords_normalized[1]:.3f})")
print(f"Кандидатів в кластері: {len(result.cluster_candidates)}")

print(f"\n--- Гіпотези З КЛАСТЕРА (топ-5) ---")
for h in result.hypotheses_in_cluster[:5]:
    print(f"  {h.name}: {h.probability:.2%}")

print(f"\n--- Гіпотези НЕ З КЛАСТЕРА (топ-5) ---")
for h in result.hypotheses_outside_cluster[:5]:
    attention = " ⚠️ УВАГА!" if h.needs_attention else ""
    print(f"  {h.name}: {h.probability:.2%}{attention}")

print(f"\nПотребує уваги діагноз не з кластера: {result.has_outside_cluster_attention}")

# Цикл питань
print("\n[3] Цикл питань (автоматичні відповіді)...")

responses = [True, False, True, False, True]  # Симуляція відповідей
i = 0

while controller.should_continue() and i < len(responses):
    q = controller.get_next_question()
    if q is None:
        break
    
    print(f"\nQ{i+1}: {q.text}")
    print(f"     Симптом: {q.symptom}, EIG: {q.eig:.4f}")
    
    result = controller.process_answer(responses[i])
    
    print(f"     Топ з кластера: ", end="")
    for h in result.hypotheses_in_cluster[:2]:
        print(f"{h.name}({h.probability:.1%}) ", end="")
    print()
    
    if result.hypotheses_outside_cluster:
        print(f"     Топ НЕ з кластера: ", end="")
        for h in result.hypotheses_outside_cluster[:2]:
            attn = "⚠️" if h.needs_attention else ""
            print(f"{h.name}({h.probability:.1%}){attn} ", end="")
        print()
    
    i += 1

# Фінальний результат
print("\n[4] Фінальний результат...")
final = controller.get_result()

print(f"\nПричина зупинки: {final.stop_reason}")
print(f"Ітерацій: {final.iterations}")
print(f"Питань: {final.questions_asked}")

print(f"\n--- Топ діагнози ---")
for name, prob in final.top_hypotheses[:5]:
    print(f"  {name}: {prob:.2%}")

print(f"\n--- З кластера ---")
for name, prob in final.top_in_cluster[:3]:
    print(f"  {name}: {prob:.2%}")

print(f"\n--- НЕ з кластера ---")
for name, prob in final.top_outside_cluster[:3]:
    print(f"  {name}: {prob:.2%}")

if final.needs_additional_investigation:
    print(f"\n⚠️ {final.outside_cluster_warning}")
else:
    print(f"\n✅ Додаткові дослідження не потрібні")

print("\n" + "=" * 60)
print("Тест завершено!")
print("=" * 60)
