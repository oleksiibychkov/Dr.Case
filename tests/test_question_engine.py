"""
Тести для модуля question_engine

Запуск: pytest tests/test_question_engine.py -v
Або демо: python tests/test_question_engine.py
"""

from pathlib import Path
import numpy as np

# Шлях до бази даних
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"


def test_information_gain_calculator():
    """Тест InformationGainCalculator"""
    from dr_case.question_engine import InformationGainCalculator
    from dr_case.encoding import DiseaseEncoder
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    # Завантажуємо дані
    encoder = DiseaseEncoder.from_database(str(DATA_PATH))
    matrix = encoder.encode_all(normalize=False)
    
    calculator = InformationGainCalculator(
        matrix,
        encoder.disease_names,
        encoder.vocabulary.symptoms
    )
    
    print(f"✓ InformationGainCalculator created")
    print(f"  Diseases: {len(encoder.disease_names)}")
    print(f"  Symptoms: {len(encoder.vocabulary.symptoms)}")
    
    # Тест для підмножини кандидатів
    candidates = encoder.disease_names[:20]  # Перші 20
    
    results = calculator.compute_all(
        candidates=candidates,
        top_n=5
    )
    
    assert len(results) > 0
    assert results[0].information_gain >= results[-1].information_gain  # Відсортовано
    
    print(f"\n✓ Information Gain test (top 5 for 20 candidates):")
    for r in results:
        print(f"  {r.symptom}: IG={r.information_gain:.4f}, split={r.split_ratio:.2%}")
    
    return calculator


def test_question_generator_creation():
    """Тест створення QuestionGenerator"""
    from dr_case.question_engine import QuestionGenerator, QuestionStrategy
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(
        str(DATA_PATH),
        strategy=QuestionStrategy.HYBRID
    )
    
    print(f"✓ QuestionGenerator created: {generator}")
    
    return generator


def test_generate_questions():
    """Тест генерації питань"""
    from dr_case.question_engine import QuestionGenerator
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    # Симулюємо кандидатів (респіраторні захворювання)
    candidates = [
        "Influenza",
        "Common Cold", 
        "Bronchitis",
        "Pneumonia",
        "Bronchial Asthma",
        "Chronic Obstructive Pulmonary Disease (Copd)",
    ]
    
    known_symptoms = ["fever", "cough"]
    
    questions = generator.generate(
        candidates=candidates,
        known_present=known_symptoms,
        n_questions=5
    )
    
    assert len(questions) > 0
    
    print(f"\n✓ Generated {len(questions)} questions")
    print(f"  Candidates: {candidates}")
    print(f"  Known symptoms: {known_symptoms}")
    print(f"\n  Questions:")
    
    for i, q in enumerate(questions):
        print(f"    {i+1}. {q.text}")
        print(f"       IG: {q.information_gain:.4f}")
    
    return questions


def test_generate_with_absent():
    """Тест генерації з відсутніми симптомами"""
    from dr_case.question_engine import QuestionGenerator
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    candidates = [
        "Influenza",
        "Common Cold",
        "Allergic Rhinitis",
        "Sinusitis",
    ]
    
    present = ["headache", "runny nose"]
    absent = ["fever", "cough"]  # Виключаємо ці симптоми
    
    questions = generator.generate(
        candidates=candidates,
        known_present=present,
        known_absent=absent,
        n_questions=3
    )
    
    # Перевіряємо що fever і cough не пропонуються
    for q in questions:
        assert q.symptom not in absent, f"Should not ask about known absent: {q.symptom}"
        assert q.symptom not in present, f"Should not ask about known present: {q.symptom}"
    
    print(f"\n✓ Questions with absent symptoms test")
    print(f"  Present: {present}")
    print(f"  Absent: {absent}")
    print(f"  Questions:")
    for q in questions:
        print(f"    - {q.symptom} (IG={q.information_gain:.4f})")
    
    return questions


def test_strategies():
    """Тест різних стратегій"""
    from dr_case.question_engine import QuestionGenerator, QuestionStrategy
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    candidates = [
        "Influenza",
        "Pneumonia", 
        "Bronchitis",
        "Tuberculosis",
        "Lung Cancer",
    ]
    
    known = ["cough", "fatigue"]
    
    print(f"\n✓ Strategy comparison test")
    print(f"  Candidates: {candidates}")
    
    for strategy in QuestionStrategy:
        questions = generator.generate(
            candidates=candidates,
            known_present=known,
            n_questions=3,
            strategy=strategy
        )
        
        print(f"\n  {strategy.value}:")
        for q in questions[:3]:
            print(f"    - {q.symptom} (IG={q.information_gain:.4f})")
    
    return True


def test_single_question():
    """Тест генерації одного питання"""
    from dr_case.question_engine import QuestionGenerator
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    candidates = ["Influenza", "Common Cold", "Bronchitis"]
    
    question = generator.generate_single(
        candidates=candidates,
        known_present=["fever"]
    )
    
    assert question is not None
    assert question.symptom not in ["fever"]
    
    print(f"\n✓ Single question: {question.text}")
    print(f"  Symptom: {question.symptom}")
    print(f"  IG: {question.information_gain:.4f}")
    
    return question


def test_explain_question():
    """Тест пояснення питання"""
    from dr_case.question_engine import QuestionGenerator
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    candidates = ["Influenza", "Common Cold", "Pneumonia", "Bronchitis"]
    
    question = generator.generate_single(
        candidates=candidates,
        known_present=["fever", "cough"]
    )
    
    if question:
        explanation = generator.explain_question(question, candidates)
        
        print(f"\n✓ Question explanation:")
        print(explanation)
    
    return True


def test_common_symptoms():
    """Тест знаходження спільних симптомів"""
    from dr_case.question_engine import QuestionGenerator
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    candidates = ["Influenza", "Common Cold", "Bronchitis"]
    
    common = generator.get_common_symptoms(candidates, min_coverage=0.6)
    
    print(f"\n✓ Common symptoms (60%+ coverage):")
    print(f"  Candidates: {candidates}")
    print(f"  Common symptoms: {common[:10]}")
    
    return common


def test_distinguishing_symptoms():
    """Тест знаходження відмінних симптомів"""
    from dr_case.question_engine import QuestionGenerator
    
    if not DATA_PATH.exists():
        print(f"⚠ Skipping: {DATA_PATH} not found")
        return None
    
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    disease1 = "Influenza"
    disease2 = "Common Cold"
    
    only_in_1, only_in_2 = generator.get_distinguishing_symptoms(disease1, disease2)
    
    print(f"\n✓ Distinguishing symptoms:")
    print(f"  {disease1} vs {disease2}")
    print(f"  Only in {disease1}: {only_in_1[:5]}")
    print(f"  Only in {disease2}: {only_in_2[:5]}")
    
    return only_in_1, only_in_2


def test_with_candidate_selector():
    """Тест інтеграції з CandidateSelector"""
    from dr_case.question_engine import QuestionGenerator
    from dr_case.candidate_selector import CandidateSelector
    
    MODEL_PATH = Path(__file__).parent.parent / "models" / "som_optimized.pkl"
    
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        print(f"⚠ Skipping integration test: files not found")
        return None
    
    # Завантажуємо CandidateSelector
    selector = CandidateSelector.from_model_file(
        str(MODEL_PATH),
        str(DATA_PATH)
    )
    
    # Створюємо QuestionGenerator
    generator = QuestionGenerator.from_database(str(DATA_PATH))
    
    # Симптоми пацієнта
    symptoms = ["fever", "headache", "cough"]
    
    # Отримуємо кандидатів
    selection = selector.select(symptoms)
    
    # Генеруємо питання
    questions = generator.generate(
        candidates=selection.candidates,
        known_present=symptoms,
        n_questions=5
    )
    
    print(f"\n✓ Integration with CandidateSelector")
    print(f"  Input symptoms: {symptoms}")
    print(f"  Candidates: {selection.candidate_count}")
    print(f"  Generated questions:")
    
    for i, q in enumerate(questions):
        print(f"    {i+1}. {q.text} (IG={q.information_gain:.4f})")
    
    return questions


def demo():
    """Повна демонстрація модуля question_engine"""
    print("=" * 60)
    print("Dr.Case — Демонстрація Question Engine")
    print("=" * 60)
    
    if not DATA_PATH.exists():
        print(f"\n❌ ПОМИЛКА: База даних не знайдена!")
        print(f"   Очікуваний шлях: {DATA_PATH}")
        return False
    
    try:
        print("\n--- 1. Information Gain Calculator ---")
        test_information_gain_calculator()
        
        print("\n--- 2. Question Generator Creation ---")
        test_question_generator_creation()
        
        print("\n--- 3. Generate Questions ---")
        test_generate_questions()
        
        print("\n--- 4. Questions with Absent Symptoms ---")
        test_generate_with_absent()
        
        print("\n--- 5. Different Strategies ---")
        test_strategies()
        
        print("\n--- 6. Single Question ---")
        test_single_question()
        
        print("\n--- 7. Explain Question ---")
        test_explain_question()
        
        print("\n--- 8. Common Symptoms ---")
        test_common_symptoms()
        
        print("\n--- 9. Distinguishing Symptoms ---")
        test_distinguishing_symptoms()
        
        print("\n--- 10. Integration with CandidateSelector ---")
        test_with_candidate_selector()
        
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
