"""
Dr.Case — Тестування Question Engine

Демонстрація роботи:
1. Завантаження бази даних
2. Симуляція діагностики з NN
3. Вибір питань на основі EIG
4. Оновлення ймовірностей

Запуск:
    python scripts/test_question_engine.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np

# Імпорти
from dr_case.question_engine import (
    InformationGainCalculator,
    QuestionSelector,
    AnswerProcessor,
    SessionState,
    AnswerType,
)


def main():
    print("=" * 70)
    print("Dr.Case — ТЕСТУВАННЯ QUESTION ENGINE")
    print("=" * 70)
    
    # Шляхи
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if not database_path.exists():
        print(f"❌ База даних не знайдена: {database_path}")
        return
    
    # ========== ЗАВАНТАЖЕННЯ ==========
    
    print("\n🔄 Завантаження бази даних...")
    
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    print(f"   Завантажено {len(database)} хвороб")
    
    # ========== ІНІЦІАЛІЗАЦІЯ ==========
    
    print("\n🔄 Ініціалізація Question Engine...")
    
    # Information Gain Calculator
    eig_calc = InformationGainCalculator.from_database(
        database,
        min_disease_prob=0.05  # Параметр: мін. ймовірність хвороби
    )
    
    print(f"   {eig_calc}")
    
    # Question Selector
    selector = QuestionSelector.from_loaded_database(
        database,
        language="uk",
        min_disease_prob=0.05,
        min_eig_threshold=0.001
    )
    
    print(f"   {selector}")
    
    # Answer Processor
    processor = AnswerProcessor(eig_calc)
    
    # ========== СИМУЛЯЦІЯ: Сценарій 1 ==========
    
    print("\n" + "=" * 70)
    print("📋 СЦЕНАРІЙ 1: Грипоподібні симптоми")
    print("=" * 70)
    
    # Симулюємо початкові ймовірності від NN
    # (в реальності це буде результат TwoBranchNN)
    initial_probs = {
        "Influenza": 0.35,
        "Common Cold": 0.25,
        "COVID-19": 0.15,
        "Bronchitis": 0.10,
        "Pneumonia": 0.08,
        "Sinusitis": 0.05,
        "Pharyngitis": 0.02,
    }
    
    known_symptoms = {"Fever", "Cough", "Headache"}
    
    print(f"\n📊 Початкові симптоми: {known_symptoms}")
    print(f"\n📊 Початкові ймовірності від NN:")
    for disease, prob in sorted(initial_probs.items(), key=lambda x: -x[1]):
        print(f"   {disease:20s}: {prob:.1%}")
    
    # Створюємо стан сесії
    state = SessionState()
    state.known_symptoms = known_symptoms.copy()
    state.disease_probs = initial_probs.copy()
    
    # Цикл питань
    max_questions = 5
    
    for i in range(max_questions):
        print(f"\n{'─' * 50}")
        print(f"❓ ПИТАННЯ {i + 1}")
        print(f"{'─' * 50}")
        
        # Вибираємо питання
        question = selector.select_question(
            disease_probs=state.disease_probs,
            known_symptoms=state.known_symptoms,
            asked_symptoms=state.all_asked_symptoms
        )
        
        if question is None:
            print("   Більше немає корисних питань.")
            break
        
        print(f"\n   Питання: {question.text}")
        print(f"   Симптом: {question.symptom}")
        print(f"   EIG: {question.eig:.4f}")
        print(f"   P(yes): {question.p_yes:.1%}, P(no): {question.p_no:.1%}")
        
        # Пояснення
        explanation = selector.explain_question(question, state.disease_probs)
        print(f"\n   📝 Пояснення:")
        for line in explanation.split('\n'):
            print(f"      {line}")
        
        # Симулюємо відповідь (YES для демонстрації)
        # В реальності — від користувача
        if i == 0:
            answer = AnswerType.YES  # Muscle Pain = YES (типово для грипу)
        elif i == 1:
            answer = AnswerType.NO   # Loss of Smell = NO (не COVID)
        elif i == 2:
            answer = AnswerType.YES
        else:
            answer = AnswerType.UNKNOWN
        
        print(f"\n   👤 Відповідь: {answer.value}")
        
        # Обробляємо відповідь
        state = processor.process_answer(state, question.symptom, answer)
        
        # Показуємо оновлені ймовірності
        print(f"\n   📊 Оновлені ймовірності:")
        for disease, prob in sorted(state.disease_probs.items(), key=lambda x: -x[1])[:5]:
            print(f"      {disease:20s}: {prob:.1%}")
    
    # ========== ПІДСУМОК ==========
    
    print("\n" + "=" * 70)
    print("📊 ПІДСУМОК СЕСІЇ")
    print("=" * 70)
    
    print(f"\n   Питань задано: {state.questions_asked}")
    print(f"   Підтверджених симптомів: {state.known_symptoms}")
    print(f"   Заперечених симптомів: {state.negated_symptoms}")
    print(f"   'Не знаю': {state.unknown_symptoms}")
    
    print(f"\n   🎯 Топ-5 діагнозів:")
    top_5 = processor.get_diagnosis_summary(state, top_n=5)
    for disease, prob in top_5:
        print(f"      {disease:20s}: {prob:.1%}")
    
    # ========== ТЕСТ EIG ==========
    
    print("\n" + "=" * 70)
    print("🔬 ТЕСТ: Обчислення EIG для різних симптомів")
    print("=" * 70)
    
    test_probs = {
        "Influenza": 0.40,
        "Common Cold": 0.30,
        "COVID-19": 0.20,
        "Bronchitis": 0.10,
    }
    
    print(f"\n   Тестові ймовірності:")
    for d, p in test_probs.items():
        print(f"      {d}: {p:.0%}")
    
    # Обчислюємо EIG для кількох симптомів
    test_symptoms = ["Muscle Pain", "Loss of Smell", "Sore Throat", "Fatigue", "Chills"]
    
    print(f"\n   {'Симптом':<20} {'EIG':>8} {'P(yes)':>8} {'P(no)':>8}")
    print("   " + "-" * 50)
    
    for symptom in test_symptoms:
        result = eig_calc.compute_eig(symptom, test_probs)
        print(f"   {symptom:<20} {result.eig:>8.4f} {result.p_yes:>8.1%} {result.p_no:>8.1%}")
    
    # Топ-10 питань
    print(f"\n   🔝 Топ-10 питань за EIG:")
    top_questions = selector.select_top_questions(test_probs, top_k=10)
    
    for i, q in enumerate(top_questions, 1):
        print(f"      {i:2d}. {q.symptom:<25} EIG={q.eig:.4f}")
    
    # ========== ТЕСТ ДИСКРИМІНАЦІЇ ==========
    
    print("\n" + "=" * 70)
    print("🔬 ТЕСТ: Симптоми що розрізняють хвороби")
    print("=" * 70)
    
    pairs = [
        ("Influenza", "Common Cold"),
        ("Influenza", "COVID-19"),
        ("COVID-19", "Common Cold"),
    ]
    
    for d1, d2 in pairs:
        print(f"\n   {d1} vs {d2}:")
        discriminative = eig_calc.get_discriminative_symptoms(d1, d2, top_k=3)
        for symptom, f1, f2 in discriminative:
            print(f"      {symptom:<20}: {f1:.0%} vs {f2:.0%} (diff={abs(f1-f2):.0%})")
    
    print("\n" + "=" * 70)
    print("✅ ТЕСТУВАННЯ ЗАВЕРШЕНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
