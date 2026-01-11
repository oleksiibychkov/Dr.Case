"""
Тести для модуля diagnosis_engine

Запуск: pytest tests/test_diagnosis_engine.py -v
Або демо: python tests/test_diagnosis_engine.py
"""

from pathlib import Path
from typing import Dict

# Шляхи
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"
SOM_MODEL_PATH = Path(__file__).parent.parent / "models" / "som_optimized.pkl"
NN_MODEL_PATH = Path(__file__).parent.parent / "models" / "nn_model.pt"


def test_session_creation():
    """Тест створення сесії"""
    from dr_case.diagnosis_engine import DiagnosisSession, SessionStatus
    
    session = DiagnosisSession()
    
    assert session.status == SessionStatus.ACTIVE
    assert session.current_cycle == 0
    assert len(session.present_symptoms) == 0
    
    print(f"✓ Session created: {session.session_id}")
    
    return session


def test_session_symptoms():
    """Тест роботи з симптомами"""
    from dr_case.diagnosis_engine import DiagnosisSession
    
    session = DiagnosisSession()
    
    # Додаємо симптоми
    session.add_symptom("fever", present=True)
    session.add_symptom("cough", present=True)
    session.add_symptom("rash", present=False)
    
    assert "fever" in session.present_symptoms
    assert "cough" in session.present_symptoms
    assert "rash" in session.absent_symptoms
    
    # Змінюємо статус симптому
    session.add_symptom("fever", present=False)
    assert "fever" in session.absent_symptoms
    assert "fever" not in session.present_symptoms
    
    print(f"✓ Session symptoms test")
    print(f"  Present: {session.present_symptoms}")
    print(f"  Absent: {session.absent_symptoms}")
    
    return session


def test_engine_creation():
    """Тест створення DiagnosisEngine"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    if not DATA_PATH.exists() or not SOM_MODEL_PATH.exists():
        print(f"⚠ Skipping: required files not found")
        return None
    
    engine = DiagnosisEngine.from_models(
        som_model_path=str(SOM_MODEL_PATH),
        database_path=str(DATA_PATH),
        nn_model_path=str(NN_MODEL_PATH) if NN_MODEL_PATH.exists() else None
    )
    
    print(f"✓ DiagnosisEngine created: {engine}")
    print(f"  Has NN: {engine.nn_trainer is not None}")
    
    return engine


def test_quick_diagnosis():
    """Тест швидкої діагностики"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    symptoms = ["fever", "cough", "headache", "fatigue"]
    
    result = engine.diagnose_quick(symptoms)
    
    assert len(result.hypotheses) > 0
    assert result.top_diagnosis is not None
    
    print(f"\n✓ Quick diagnosis test")
    print(f"  Symptoms: {symptoms}")
    print(f"  Candidates evaluated: {len(result.hypotheses)}")
    print(f"  Top 5 diagnoses:")
    
    for i, h in enumerate(result.get_top_n(5)):
        print(f"    {i+1}. {h.disease_name}: {h.confidence:.1%}")
    
    return result


def test_quick_diagnosis_with_absent():
    """Тест швидкої діагностики з відсутніми симптомами"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    present = ["fever", "cough", "shortness of breath"]
    absent = ["rash", "vomiting", "diarrhea"]
    
    result = engine.diagnose_quick(present, absent)
    
    print(f"\n✓ Quick diagnosis with absent symptoms")
    print(f"  Present: {present}")
    print(f"  Absent: {absent}")
    print(f"  Top diagnosis: {result.top_diagnosis.disease_name}")
    
    return result


def test_start_session():
    """Тест початку сесії"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    symptoms = ["fever", "headache"]
    
    session = engine.start_session(symptoms)
    
    assert session.is_active
    assert len(session.current_candidates) > 0
    assert session.top_diagnosis is not None
    
    print(f"\n✓ Session started")
    print(f"  Session ID: {session.session_id}")
    print(f"  Initial symptoms: {session.present_symptoms}")
    print(f"  Candidates: {session.n_candidates}")
    print(f"  Top diagnosis: {session.top_diagnosis.disease_name} ({session.top_confidence:.1%})")
    
    return session


def test_get_questions():
    """Тест отримання питань"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    session = engine.start_session(["fever", "cough"])
    
    # Отримуємо питання
    questions = engine.get_next_questions(session, n=3)
    
    assert len(questions) > 0
    
    print(f"\n✓ Questions generated")
    print(f"  Current candidates: {session.n_candidates}")
    print(f"  Questions:")
    
    for i, q in enumerate(questions):
        print(f"    {i+1}. {q.text} (IG={q.information_gain:.4f})")
    
    return questions


def test_process_answer():
    """Тест обробки відповіді"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    session = engine.start_session(["fever", "cough"])
    
    candidates_before = session.n_candidates
    
    # Отримуємо питання
    question = engine.get_next_question(session)
    
    if question:
        print(f"\n✓ Processing answer test")
        print(f"  Question: {question.text}")
        print(f"  Candidates before: {candidates_before}")
        
        # Відповідаємо "так"
        engine.process_answer(session, question, answer=True)
        
        print(f"  Answer: Yes")
        print(f"  New symptom: {question.symptom} (present)")
        print(f"  Candidates after: {session.n_candidates}")
        print(f"  Top diagnosis: {session.top_diagnosis.disease_name} ({session.top_confidence:.1%})")
    
    return session


def test_interactive_simulation():
    """Тест симуляції інтерактивної діагностики"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    # Симулюємо пацієнта з грипом
    # Ці симптоми будуть "справжніми" для пацієнта
    patient_true_symptoms = {
        "fever": True,
        "headache": True,
        "cough": True,
        "fatigue": True,
        "body_aches": True,
        "chills": True,
        "runny nose": True,
        "rash": False,
        "vomiting": False,
        "diarrhea": False,
        "chest_pain": False,
    }
    
    # Функція для відповідей
    def answer_func(question):
        symptom = question.symptom
        # Якщо знаємо відповідь - даємо її, інакше випадково
        if symptom in patient_true_symptoms:
            return patient_true_symptoms[symptom]
        return False  # За замовчуванням - ні
    
    print(f"\n✓ Interactive simulation test")
    print(f"  Simulating patient with flu-like symptoms")
    
    # Початкові симптоми
    initial = ["fever", "cough"]
    
    session = engine.start_session(initial)
    print(f"\n  Initial: {initial}")
    print(f"  Candidates: {session.n_candidates}")
    
    # Запускаємо цикли
    cycle_count = 0
    while session.should_continue and cycle_count < 3:
        cycle_count += 1
        print(f"\n  --- Cycle {cycle_count} ---")
        
        # Отримуємо питання
        questions = engine.get_next_questions(session, n=2)
        
        for q in questions:
            if not session.should_continue:
                break
            
            answer = answer_func(q)
            engine.process_answer(session, q, answer)
            
            answer_str = "Yes" if answer else "No"
            print(f"    Q: {q.symptom}? → {answer_str}")
        
        print(f"    Top: {session.top_diagnosis.disease_name} ({session.top_confidence:.1%})")
        print(f"    Candidates: {session.n_candidates}")
    
    # Результат
    result = engine.get_result(session)
    
    print(f"\n  === Final Result ===")
    print(f"  Questions asked: {result.questions_asked}")
    print(f"  Confidence: {result.confidence_level.value}")
    print(f"  Top diagnosis: {result.top_diagnosis.disease_name}")
    
    return result


def test_explain_diagnosis():
    """Тест пояснення діагнозу"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    session = engine.start_session(["fever", "cough", "headache"])
    
    # Додаємо кілька відповідей
    for _ in range(2):
        q = engine.get_next_question(session)
        if q:
            engine.process_answer(session, q, answer=True)
    
    # Пояснення
    explanation = engine.explain_diagnosis(session)
    
    print(f"\n✓ Diagnosis explanation:")
    print(explanation)
    
    return explanation


def test_session_summary():
    """Тест підсумку сесії"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    engine = _get_engine()
    if engine is None:
        return None
    
    session = engine.start_session(["fever", "cough"])
    
    # Задаємо кілька питань
    for _ in range(3):
        q = engine.get_next_question(session)
        if q:
            engine.process_answer(session, q, answer=True)
    
    session.complete()
    
    summary = session.get_summary()
    
    print(f"\n✓ Session summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary


def _get_engine():
    """Отримати engine для тестів"""
    from dr_case.diagnosis_engine import DiagnosisEngine
    
    if not DATA_PATH.exists() or not SOM_MODEL_PATH.exists():
        print(f"⚠ Required files not found")
        return None
    
    return DiagnosisEngine.from_models(
        som_model_path=str(SOM_MODEL_PATH),
        database_path=str(DATA_PATH),
        nn_model_path=str(NN_MODEL_PATH) if NN_MODEL_PATH.exists() else None
    )


def demo():
    """Повна демонстрація модуля diagnosis_engine"""
    print("=" * 60)
    print("Dr.Case — Демонстрація Diagnosis Engine")
    print("=" * 60)
    
    if not DATA_PATH.exists():
        print(f"\n❌ ПОМИЛКА: База даних не знайдена!")
        print(f"   Очікуваний шлях: {DATA_PATH}")
        return False
    
    if not SOM_MODEL_PATH.exists():
        print(f"\n❌ ПОМИЛКА: SOM модель не знайдена!")
        print(f"   Очікуваний шлях: {SOM_MODEL_PATH}")
        return False
    
    try:
        print("\n--- 1. Session Creation ---")
        test_session_creation()
        
        print("\n--- 2. Session Symptoms ---")
        test_session_symptoms()
        
        print("\n--- 3. Engine Creation ---")
        test_engine_creation()
        
        print("\n--- 4. Quick Diagnosis ---")
        test_quick_diagnosis()
        
        print("\n--- 5. Quick Diagnosis with Absent ---")
        test_quick_diagnosis_with_absent()
        
        print("\n--- 6. Start Session ---")
        test_start_session()
        
        print("\n--- 7. Get Questions ---")
        test_get_questions()
        
        print("\n--- 8. Process Answer ---")
        test_process_answer()
        
        print("\n--- 9. Interactive Simulation ---")
        test_interactive_simulation()
        
        print("\n--- 10. Explain Diagnosis ---")
        test_explain_diagnosis()
        
        print("\n--- 11. Session Summary ---")
        test_session_summary()
        
        print("\n" + "=" * 60)
        print("✅ Всі тести пройдено успішно!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo()
