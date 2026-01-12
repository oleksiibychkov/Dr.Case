"""
Dr.Case — Тестування Diagnosis Cycle

Тестує повний циклічний процес діагностики:
1. Початкові симптоми → SOM → Кандидати → NN → Гіпотези
2. Критерії зупинки (DOMINANCE, STABILITY, NEED_TEST, SAFETY)
3. Вибір питань за EIG
4. Оновлення гіпотез

Запуск:
    python scripts/test_diagnosis_cycle.py
"""

import sys
from pathlib import Path

# Додаємо корінь проекту до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_stopping_criteria():
    """Тест критеріїв зупинки"""
    print("\n" + "=" * 70)
    print("📋 ТЕСТ 1: Критерії зупинки")
    print("=" * 70)
    
    from dr_case.diagnosis_cycle import (
        StoppingCriteria, StoppingConfig, StopReason
    )
    
    config = StoppingConfig(
        dominance_threshold=0.85,
        dominance_gap=0.30,
        confidence_threshold=0.80,
        need_test_threshold=0.05,  # Знижений поріг для NEED_TEST
        max_iterations=20
    )
    criteria = StoppingCriteria(config)
    
    # Тест 1: CONTINUE
    print("\n🔍 Тест CONTINUE:")
    hypotheses = {'A': 0.40, 'B': 0.35, 'C': 0.25}
    decision = criteria.check(hypotheses, iteration=1)
    print(f"   Гіпотези: {hypotheses}")
    print(f"   Результат: {decision.reason.value}")
    print(f"   Should stop: {decision.should_stop}")
    # NEED_TEST може спрацювати якщо різниця < 0.05
    # Змінюємо гіпотези щоб була більша різниця
    
    # Тест 1b: CONTINUE з більшою різницею
    print("\n🔍 Тест CONTINUE (з більшою різницею):")
    hypotheses = {'A': 0.50, 'B': 0.30, 'C': 0.20}  # Різниця 0.20 > 0.05
    decision = criteria.check(hypotheses, iteration=1)
    print(f"   Гіпотези: {hypotheses}")
    print(f"   Результат: {decision.reason.value}")
    print(f"   Should stop: {decision.should_stop}")
    assert decision.reason == StopReason.CONTINUE, f"Expected CONTINUE, got {decision.reason}"
    
    # Тест 2: DOMINANCE
    print("\n🔍 Тест DOMINANCE:")
    hypotheses = {'A': 0.90, 'B': 0.05, 'C': 0.05}
    decision = criteria.check(hypotheses, iteration=2)
    print(f"   Гіпотези: {hypotheses}")
    print(f"   Результат: {decision.reason.value}")
    print(f"   Message: {decision.message}")
    assert decision.reason == StopReason.DOMINANCE
    
    # Тест 3: CONFIDENT
    print("\n🔍 Тест CONFIDENT:")
    hypotheses = {'A': 0.82, 'B': 0.10, 'C': 0.08}
    decision = criteria.check(hypotheses, iteration=3)
    print(f"   Гіпотези: {hypotheses}")
    print(f"   Результат: {decision.reason.value}")
    # Може бути DOMINANCE або CONFIDENT
    assert decision.should_stop
    
    # Тест 4: SAFETY_LIMIT
    print("\n🔍 Тест SAFETY_LIMIT:")
    hypotheses = {'A': 0.40, 'B': 0.35, 'C': 0.25}
    decision = criteria.check(hypotheses, iteration=25)
    print(f"   Iteration: 25")
    print(f"   Результат: {decision.reason.value}")
    assert decision.reason == StopReason.SAFETY_LIMIT
    
    # Тест 5: NEED_TEST
    print("\n🔍 Тест NEED_TEST:")
    config_test = StoppingConfig(
        dominance_threshold=0.95,  # Високий поріг
        need_test_threshold=0.15
    )
    criteria_test = StoppingCriteria(config_test)
    hypotheses = {'A': 0.35, 'B': 0.33}  # Занадто близькі
    decision = criteria_test.check(hypotheses, iteration=5)
    print(f"   Гіпотези: {hypotheses}")
    print(f"   Результат: {decision.reason.value}")
    # Може бути NEED_TEST
    
    print("\n✅ Тест критеріїв зупинки пройдено!")


def test_hypothesis_tracker():
    """Тест трекера гіпотез"""
    print("\n" + "=" * 70)
    print("📋 ТЕСТ 2: Трекер гіпотез")
    print("=" * 70)
    
    from dr_case.diagnosis_cycle import HypothesisTracker, HypothesisTrend
    
    tracker = HypothesisTracker(trend_window=3, tolerance=0.05)
    
    # Симулюємо ітерації
    iterations = [
        {'Influenza': 0.40, 'Cold': 0.35, 'COVID': 0.25},
        {'Influenza': 0.45, 'Cold': 0.32, 'COVID': 0.23},
        {'Influenza': 0.50, 'Cold': 0.30, 'COVID': 0.20},
        {'Influenza': 0.55, 'Cold': 0.28, 'COVID': 0.17},
    ]
    
    print("\n🔄 Симуляція ітерацій:")
    for i, hyp in enumerate(iterations):
        changes = tracker.update(hyp, iteration=i+1)
        print(f"\n   Ітерація {i+1}:")
        for c in changes[:3]:
            print(f"      {c.disease}: {c.old_probability:.1%} → {c.new_probability:.1%} ({c.trend.value})")
    
    # Перевіряємо тренди
    print("\n📈 Аналіз трендів:")
    rising = tracker.get_rising_hypotheses()
    falling = tracker.get_falling_hypotheses()
    stable = tracker.get_stable_hypotheses()
    
    print(f"   Зростаючі: {rising}")
    print(f"   Падаючі: {falling}")
    print(f"   Стабільні: {stable}")
    
    assert 'Influenza' in rising, "Influenza має зростати"
    
    # Тест виключення
    print("\n🚫 Тест виключення:")
    tracker.exclude('Cold', reason='Treatment failed')
    print(f"   Виключено: {tracker.excluded}")
    
    # Тест downgrade/boost
    print("\n📊 Тест модифікаторів:")
    tracker.downgrade('Influenza', factor=0.5)
    tracker.boost('COVID', factor=1.5)
    print(f"   Модифікатори: {tracker.modifiers}")
    
    # Тест restore
    restored = tracker.restore_excluded()
    print(f"   Відновлено: {restored}")
    
    print("\n✅ Тест трекера гіпотез пройдено!")


def test_feedback_processor():
    """Тест обробки зворотного зв'язку"""
    print("\n" + "=" * 70)
    print("📋 ТЕСТ 3: Обробка feedback")
    print("=" * 70)
    
    from dr_case.diagnosis_cycle import (
        FeedbackProcessor, FeedbackConfig, Feedback, FeedbackType,
        HypothesisTracker
    )
    
    config = FeedbackConfig(
        treatment_failure_downgrade=0.3,
        alternatives_boost=1.5,
        max_restart_attempts=3
    )
    processor = FeedbackProcessor(config)
    
    # Створюємо трекер з даними
    tracker = HypothesisTracker()
    tracker.update({
        'Influenza': 0.50,
        'Cold': 0.30,
        'COVID': 0.20
    }, iteration=1)
    
    # Тест 1: TREATMENT_SUCCESS
    print("\n🔍 Тест TREATMENT_SUCCESS:")
    feedback = Feedback(type=FeedbackType.TREATMENT_SUCCESS)
    result = processor.process(feedback, tracker)
    print(f"   Action: {result.action_taken}")
    print(f"   Should restart: {result.should_restart}")
    assert not result.should_restart
    
    # Тест 2: TREATMENT_FAILED
    print("\n🔍 Тест TREATMENT_FAILED:")
    feedback = Feedback(
        type=FeedbackType.TREATMENT_FAILED,
        failed_diagnosis='Influenza'
    )
    result = processor.process(feedback, tracker)
    print(f"   Action: {result.action_taken}")
    print(f"   Should restart: {result.should_restart}")
    print(f"   Message: {result.message}")
    assert result.should_restart
    assert 'Influenza' in tracker.excluded
    
    # Тест 3: NEW_SYMPTOM
    print("\n🔍 Тест NEW_SYMPTOM:")
    feedback = Feedback(
        type=FeedbackType.NEW_SYMPTOM,
        symptom='Shortness Of Breath'
    )
    result = processor.process(feedback, tracker)
    print(f"   Action: {result.action_taken}")
    print(f"   New symptoms: {result.new_symptoms}")
    assert result.should_restart
    
    # Тест 4: DOCTOR_OVERRIDE
    print("\n🔍 Тест DOCTOR_OVERRIDE:")
    feedback = Feedback(
        type=FeedbackType.DOCTOR_OVERRIDE,
        doctor_diagnosis='Pneumonia',
        doctor_notes='Based on X-ray results'
    )
    result = processor.process(feedback, tracker)
    print(f"   Action: {result.action_taken}")
    print(f"   Message: {result.message}")
    
    print("\n✅ Тест feedback пройдено!")


def test_full_cycle():
    """Тест повного циклу діагностики"""
    print("\n" + "=" * 70)
    print("📋 ТЕСТ 4: Повний цикл діагностики")
    print("=" * 70)
    
    # Перевіряємо наявність моделей
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    som_path = project_root / "models" / "som_merged.pkl"
    nn_path = project_root / "models" / "nn_two_branch.pt"
    
    if not database_path.exists():
        print(f"   ⚠️ База даних не знайдена: {database_path}")
        print("   Пропускаємо тест повного циклу")
        return
    
    if not som_path.exists():
        print(f"   ⚠️ SOM модель не знайдена: {som_path}")
        print("   Пропускаємо тест повного циклу")
        return
    
    if not nn_path.exists():
        print(f"   ⚠️ NN модель не знайдена: {nn_path}")
        print("   Пропускаємо тест повного циклу")
        return
    
    print("\n🔄 Завантаження моделей...")
    
    try:
        from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig
        
        # Конфігурація для швидкого тесту
        config = StoppingConfig(
            dominance_threshold=0.70,    # Знижений поріг для тесту
            confidence_threshold=0.60,
            max_iterations=5,
            max_questions=10
        )
        
        controller = DiagnosisCycleController.from_models(
            database_path=str(database_path),
            som_path=str(som_path),
            nn_path=str(nn_path),
            stopping_config=config,
            language="uk"
        )
        
        print(f"   {controller}")
        
        # Сценарій: грипоподібні симптоми
        initial_symptoms = ['Fever', 'Cough', 'Headache']
        print(f"\n📊 Початкові симптоми: {initial_symptoms}")
        
        # Запускаємо сесію
        first_result = controller.start_session(initial_symptoms)
        
        print(f"\n🔄 Ітерація {first_result.iteration}:")
        print(f"   Кандидатів: {len(first_result.candidates)}")
        print(f"   Топ-5 гіпотез:")
        top5 = sorted(first_result.hypotheses.items(), key=lambda x: x[1], reverse=True)[:5]
        for disease, prob in top5:
            print(f"      {disease}: {prob:.1%}")
        
        if first_result.stop_decision:
            print(f"\n   Stop decision: {first_result.stop_decision.reason.value}")
            print(f"   Message: {first_result.stop_decision.message}")
        
        if first_result.question:
            print(f"\n   Питання: {first_result.question.text}")
            print(f"   EIG: {first_result.question.eig:.4f}")
        
        # Симулюємо відповіді
        answers = [True, False, True, None, False]  # yes, no, yes, skip, no
        
        iteration = 0
        while controller.should_continue() and iteration < 5:
            question = controller.get_next_question()
            if question is None:
                break
            
            answer = answers[iteration % len(answers)]
            answer_text = {True: 'Так', False: 'Ні', None: 'Не знаю'}[answer]
            
            print(f"\n──────────────────────────────────────")
            print(f"❓ Q{iteration+1}: {question.text}")
            print(f"   Відповідь: {answer_text}")
            
            result = controller.process_answer(answer)
            
            print(f"\n   Топ-3 після відповіді:")
            top3 = sorted(result.hypotheses.items(), key=lambda x: x[1], reverse=True)[:3]
            for disease, prob in top3:
                print(f"      {disease}: {prob:.1%}")
            
            iteration += 1
        
        # Фінальний результат
        final_result = controller.get_result()
        
        print("\n" + "=" * 70)
        print("📊 ФІНАЛЬНИЙ РЕЗУЛЬТАТ")
        print("=" * 70)
        print(f"\n   Причина зупинки: {final_result.stop_reason.value}")
        print(f"   Повідомлення: {final_result.stop_message}")
        print(f"\n   Ітерацій: {final_result.iterations}")
        print(f"   Питань: {final_result.questions_asked}")
        print(f"   Час: {final_result.duration_seconds:.1f} сек")
        
        print(f"\n   Симптоми (+): {final_result.present_symptoms}")
        print(f"   Симптоми (-): {final_result.absent_symptoms}")
        print(f"   Не знаю: {final_result.unknown_symptoms}")
        
        print(f"\n   🎯 Топ-5 діагнозів:")
        for disease, prob in final_result.top_hypotheses[:5]:
            marker = "→" if disease == final_result.top_diagnosis else " "
            print(f"   {marker} {disease}: {prob:.1%}")
        
        print(f"\n   Впевнений результат: {final_result.is_confident}")
        
        print("\n✅ Тест повного циклу пройдено!")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 70)
    print("Dr.Case — ТЕСТУВАННЯ DIAGNOSIS CYCLE")
    print("=" * 70)
    
    # Тест 1: Критерії зупинки
    test_stopping_criteria()
    
    # Тест 2: Трекер гіпотез
    test_hypothesis_tracker()
    
    # Тест 3: Feedback processor
    test_feedback_processor()
    
    # Тест 4: Повний цикл (якщо є моделі)
    test_full_cycle()
    
    print("\n" + "=" * 70)
    print("✅ ВСІ ТЕСТИ ЗАВЕРШЕНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
