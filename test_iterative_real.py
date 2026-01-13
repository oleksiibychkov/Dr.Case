"""
Dr.Case — Ітераційний тест повного циклу діагностики

Тестує:
1. Повний цикл з різними сценаріями
2. Питання та відповіді
3. Критерії зупинки
4. Якість діагностики
"""

from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig
from typing import List, Dict
import random

print("=" * 60)
print("ІТЕРАЦІЙНИЙ ТЕСТ Dr.Case")
print("=" * 60)

# Конфігурація
config = StoppingConfig(
    dominance_threshold=0.85,
    dominance_gap=0.30,
    max_iterations=10,
    max_questions=15
)

# Завантаження
print("\n[1] Завантаження моделей...")
controller = DiagnosisCycleController.from_models(
    database_path='data/database_lowercase.json',
    som_path='models/som_real.pkl',
    nn_path='models/nn_real.pt',
    stopping_config=config
)

# ============================================================
# ТЕСТОВІ СЦЕНАРІЇ
# ============================================================

test_scenarios = [
    {
        "name": "Грип (класичний)",
        "initial_symptoms": ["fever", "cough", "muscle pain", "fatigue"],
        "additional_answers": {
            "chills": True,
            "headache": True,
            "sore throat": True,
            "runny nose": False,
            "loss of smell": False,
        },
        "expected_top": ["influenza", "covid-19", "common cold"],
    },
    {
        "name": "COVID-19 (втрата нюху)",
        "initial_symptoms": ["fever", "cough", "loss of smell"],
        "additional_answers": {
            "fatigue": True,
            "headache": True,
            "sore throat": True,
            "muscle pain": True,
            "runny nose": False,
        },
        "expected_top": ["covid-19", "influenza"],
    },
    {
        "name": "Мігрень",
        "initial_symptoms": ["headache", "nausea"],
        "additional_answers": {
            "sensitivity to light": True,
            "vomiting": True,
            "fever": False,
            "neck pain": False,
        },
        "expected_top": ["migraine", "tension headache"],
    },
    {
        "name": "Застуда",
        "initial_symptoms": ["runny nose", "sneezing", "sore throat"],
        "additional_answers": {
            "cough": True,
            "fever": False,
            "muscle pain": False,
            "headache": False,
        },
        "expected_top": ["common cold", "allergic rhinitis"],
    },
    {
        "name": "Менінгіт (небезпечний)",
        "initial_symptoms": ["fever", "headache", "neck pain"],
        "additional_answers": {
            "nausea": True,
            "vomiting": True,
            "sensitivity to light": True,
            "confusion": True,
        },
        "expected_top": ["meningitis"],
    },
]

# ============================================================
# ЗАПУСК ТЕСТІВ
# ============================================================

results = []

for i, scenario in enumerate(test_scenarios):
    print(f"\n{'='*60}")
    print(f"СЦЕНАРІЙ {i+1}: {scenario['name']}")
    print(f"{'='*60}")
    
    print(f"Початкові симптоми: {', '.join(scenario['initial_symptoms'])}")
    
    # Старт сесії
    result = controller.start_session(scenario['initial_symptoms'])
    
    print(f"\n--- Ітерація {result.iteration} ---")
    print(f"BMU: {result.bmu_coords}")
    print(f"Кандидатів: {len(result.cluster_candidates)}")
    
    print(f"Топ-3 з кластера:")
    for h in result.hypotheses_in_cluster[:3]:
        print(f"  {h.name}: {h.probability:.1%}")
    
    if result.hypotheses_outside_cluster:
        print(f"Топ-3 НЕ з кластера:")
        for h in result.hypotheses_outside_cluster[:3]:
            attn = " ⚠️" if h.needs_attention else ""
            print(f"  {h.name}: {h.probability:.1%}{attn}")
    
    # Цикл питань
    questions_asked = 0
    max_questions = 5
    
    while controller.should_continue() and questions_asked < max_questions:
        question = controller.get_next_question()
        if question is None:
            break
        
        # Отримуємо відповідь зі сценарію або випадково
        symptom = question.symptom
        if symptom in scenario['additional_answers']:
            answer = scenario['additional_answers'][symptom]
        else:
            answer = random.choice([True, False])
        
        answer_str = "Так" if answer else "Ні"
        print(f"\nQ{questions_asked+1}: {question.text}")
        print(f"   Відповідь: {answer_str}")
        
        result = controller.process_answer(answer)
        questions_asked += 1
        
        # Показати зміни
        print(f"   Топ: {result.hypotheses_in_cluster[0].name if result.hypotheses_in_cluster else 'N/A'} " +
              f"({result.hypotheses_in_cluster[0].probability:.1%})" if result.hypotheses_in_cluster else "")
    
    # Фінальний результат
    final_result = controller.get_result()
    
    print(f"\n--- Фінальний результат ---")
    print(f"Причина зупинки: {final_result.stop_reason}")
    print(f"Ітерацій: {final_result.iterations}")
    print(f"Питань: {final_result.questions_asked}")
    
    # Топ діагнози
    print(f"\nТоп-5 діагнозів:")
    all_hypotheses = sorted(
        result.hypotheses_in_cluster + result.hypotheses_outside_cluster,
        key=lambda h: h.probability,
        reverse=True
    )[:5]
    
    for h in all_hypotheses:
        cluster_status = "✓" if h.in_cluster else "⚠️"
        print(f"  {cluster_status} {h.name}: {h.probability:.1%}")
    
    # Перевірка очікуваного
    top_names = [h.name for h in all_hypotheses[:3]]
    found_expected = any(exp in top_names for exp in scenario['expected_top'])
    
    status = "✅ PASSED" if found_expected else "❌ FAILED"
    print(f"\nОчікувано: {scenario['expected_top']}")
    print(f"Результат: {status}")
    
    results.append({
        "name": scenario['name'],
        "passed": found_expected,
        "top_diagnosis": all_hypotheses[0].name if all_hypotheses else "N/A",
        "questions": questions_asked,
    })

# ============================================================
# ПІДСУМОК
# ============================================================

print("\n" + "=" * 60)
print("ПІДСУМОК ТЕСТУВАННЯ")
print("=" * 60)

passed = sum(1 for r in results if r['passed'])
total = len(results)

print(f"\nПройдено: {passed}/{total} ({passed/total:.0%})")
print("\nДеталі:")
for r in results:
    status = "✅" if r['passed'] else "❌"
    print(f"  {status} {r['name']}: {r['top_diagnosis']} ({r['questions']} питань)")

print("\n" + "=" * 60)
