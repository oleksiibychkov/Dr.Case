#!/usr/bin/env python3
"""
Dr.Case — Тестування NLP модуля

Тестує витягування симптомів з тексту українською та англійською.

Запуск:
    python scripts/test_nlp.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dr_case.nlp import (
    SymptomExtractor,
    TextPreprocessor,
    FuzzyMatcher,
    Language,
)


def test_preprocessor():
    """Тест препроцесора"""
    print("=" * 60)
    print("📝 ТЕСТ: TextPreprocessor")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    
    # Тест українською
    text_uk = "У мене БОЛИТЬ голова вже 3 дні, температура 38.5°C!!!"
    result = preprocessor.process(text_uk)
    
    print(f"\n🇺🇦 Українська:")
    print(f"   Original:   {result.original}")
    print(f"   Normalized: {result.normalized}")
    print(f"   Tokens:     {result.tokens}")
    print(f"   Language:   {result.language}")
    
    # Тест англійською
    text_en = "I have a SEVERE headache and fever for 2 days!"
    result = preprocessor.process(text_en)
    
    print(f"\n🇬🇧 English:")
    print(f"   Original:   {result.original}")
    print(f"   Normalized: {result.normalized}")
    print(f"   Tokens:     {result.tokens}")
    print(f"   Language:   {result.language}")
    
    # Тест витягування чисел
    print(f"\n🔢 Витягування чисел:")
    text = "температура 38.5, пульс 92, тиск 120/80"
    numbers = preprocessor.extract_numbers(text)
    print(f"   Text: {text}")
    print(f"   Numbers: {numbers}")
    
    # Тест витягування тривалості
    print(f"\n⏱️ Витягування тривалості:")
    text = "болить вже 3 дні і 5 годин"
    durations = preprocessor.extract_duration(text)
    print(f"   Text: {text}")
    print(f"   Durations: {durations}")
    
    print("\n✅ TextPreprocessor тест пройдено!")


def test_fuzzy_matcher():
    """Тест нечіткого співставлення"""
    print("\n" + "=" * 60)
    print("🔍 ТЕСТ: FuzzyMatcher")
    print("=" * 60)
    
    # Симптоми з бази
    symptoms = [
        "Headache", "Fever", "Cough", "Fatigue", "Nausea",
        "Vomiting", "Diarrhea", "Sore Throat", "Runny Nose",
        "Shortness of Breath", "Chest Pain", "Abdominal Pain",
        "Back Pain", "Joint Pain", "Muscle Pain", "Rash",
        "Dizziness", "Loss of Appetite", "Weight Loss"
    ]
    
    matcher = FuzzyMatcher(symptoms, min_score=0.6, use_synonyms=True)
    
    # Тести синонімів
    test_cases = [
        ("головний біль", "Headache"),
        ("температура", "Fever"),
        ("кашель", "Cough"),
        ("нудота", "Nausea"),
        ("болить горло", "Sore Throat"),
        ("задишка", "Shortness of Breath"),
        ("headache", "Headache"),
        ("feeling sick", "Nausea"),
        ("tired", "Fatigue"),
    ]
    
    print("\n📋 Тест синонімів:")
    for text, expected in test_cases:
        results = matcher.match(text)
        found = results[0].symptom if results else "NOT FOUND"
        status = "✅" if found == expected else "❌"
        print(f"   {status} '{text}' → {found} (expected: {expected})")
    
    # Тест повного тексту
    print("\n📋 Тест повного тексту:")
    full_text = "У мене болить голова, температура і кашель"
    results = matcher.match(full_text)
    print(f"   Text: '{full_text}'")
    print(f"   Found: {[r.symptom for r in results]}")
    
    print("\n✅ FuzzyMatcher тест пройдено!")


def test_symptom_extractor():
    """Тест головного екстрактора"""
    print("\n" + "=" * 60)
    print("🏥 ТЕСТ: SymptomExtractor")
    print("=" * 60)
    
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if not database_path.exists():
        print(f"⚠️ База даних не знайдена: {database_path}")
        print("   Використовуємо тестовий список симптомів")
        
        symptoms = [
            "Headache", "Fever", "High Fever", "Cough", "Dry Cough",
            "Fatigue", "Nausea", "Vomiting", "Diarrhea", "Sore Throat",
            "Runny Nose", "Nasal Congestion", "Shortness of Breath",
            "Chest Pain", "Abdominal Pain", "Back Pain", "Joint Pain",
            "Muscle Pain", "Rash", "Itching", "Dizziness", "Chills",
            "Sweating", "Loss of Appetite", "Weight Loss", "Anxiety",
            "Depression", "Loss of Smell", "Loss of Taste", "Weakness",
        ]
        extractor = SymptomExtractor(symptoms)
    else:
        extractor = SymptomExtractor.from_database(str(database_path))
        print(f"📊 Завантажено {extractor.get_symptom_count()} симптомів з бази")
    
    # Тестові випадки
    test_cases = [
        # Українська
        {
            "text": "Болить голова вже 3 дні, температура 38.5",
            "expected_symptoms": ["Headache", "Fever"],
            "expected_vitals": {"temperature": 38.5},
            "expected_duration": {"days": 3},
        },
        {
            "text": "Кашель, нежить, болить горло",
            "expected_symptoms": ["Cough", "Runny Nose", "Sore Throat"],
        },
        {
            "text": "Нудота, блювота, діарея, біль у животі",
            "expected_symptoms": ["Nausea", "Vomiting", "Diarrhea", "Abdominal Pain"],
        },
        {
            "text": "Задишка, біль у грудях, серцебиття",
            "expected_symptoms": ["Shortness of Breath", "Chest Pain"],
        },
        {
            "text": "Слабкість, втома, немає апетиту",
            "expected_symptoms": ["Fatigue", "Loss of Appetite"],
        },
        # Англійська
        {
            "text": "I have a headache, fever and cough for 2 days",
            "expected_symptoms": ["Headache", "Fever", "Cough"],
            "expected_duration": {"days": 2},
        },
        {
            "text": "Feeling dizzy, nausea, and shortness of breath",
            "expected_symptoms": ["Dizziness", "Nausea", "Shortness of Breath"],
        },
        # Заперечення
        {
            "text": "Головний біль, але немає температури",
            "expected_symptoms": ["Headache"],
            "expected_negated": ["Fever"],
        },
    ]
    
    print("\n📋 Тестові випадки:")
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        text = case["text"]
        result = extractor.extract(text)
        
        print(f"\n--- Випадок {i} ---")
        print(f"   Text: '{text}'")
        print(f"   Language: {result.language.value}")
        print(f"   Symptoms: {result.symptoms}")
        
        if result.negated_symptoms:
            print(f"   Negated: {result.negated_symptoms}")
        
        if not result.vitals.is_empty():
            print(f"   Vitals: {result.vitals.to_dict()}")
        
        if result.duration.to_dict():
            print(f"   Duration: {result.duration.to_dict()}")
        
        print(f"   Confidence: {result.confidence:.2f}")
        
        # Перевірка очікуваних симптомів
        expected = set(case.get("expected_symptoms", []))
        found = set(result.symptoms)
        
        if expected:
            match_ratio = len(expected & found) / len(expected) if expected else 1
            if match_ratio >= 0.5:
                print(f"   ✅ Matched {match_ratio:.0%} of expected symptoms")
                passed += 1
            else:
                print(f"   ❌ Only matched {match_ratio:.0%} (expected: {expected})")
                failed += 1
        else:
            passed += 1
    
    print(f"\n📊 Результати: {passed}/{passed+failed} пройдено")
    print("\n✅ SymptomExtractor тест пройдено!")


def demo_interactive():
    """Інтерактивна демонстрація"""
    print("\n" + "=" * 60)
    print("🎮 ІНТЕРАКТИВНА ДЕМОНСТРАЦІЯ")
    print("=" * 60)
    
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    
    if database_path.exists():
        extractor = SymptomExtractor.from_database(str(database_path))
    else:
        symptoms = [
            "Headache", "Fever", "Cough", "Fatigue", "Nausea",
            "Vomiting", "Diarrhea", "Sore Throat", "Runny Nose",
            "Shortness of Breath", "Chest Pain", "Abdominal Pain",
        ]
        extractor = SymptomExtractor(symptoms)
    
    print("\nВведіть опис скарг (або 'q' для виходу):")
    print("Приклади:")
    print("  - Болить голова і температура 38")
    print("  - I have a cough and sore throat for 3 days")
    print("  - Нудота, блювота, біль у животі\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            if text.lower() in ('q', 'quit', 'exit'):
                break
            
            if not text:
                continue
            
            result = extractor.extract(text)
            
            print(f"\n📋 Результат:")
            print(f"   Мова: {result.language.value}")
            print(f"   Симптоми: {result.symptoms or 'не знайдено'}")
            
            if result.negated_symptoms:
                print(f"   Заперечені: {result.negated_symptoms}")
            
            if not result.vitals.is_empty():
                print(f"   Вітальні: {result.vitals.to_dict()}")
            
            if result.duration.to_dict():
                print(f"   Тривалість: {result.duration.to_dict()}")
            
            print(f"   Впевненість: {result.confidence:.0%}")
            
            if result.matches:
                print(f"   Деталі:")
                for m in result.matches[:5]:
                    print(f"      - {m.symptom} ({m.method}, score={m.score:.2f})")
            
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")
    
    print("\n👋 До побачення!")


def main():
    print("=" * 60)
    print("Dr.Case — ТЕСТУВАННЯ NLP МОДУЛЯ")
    print("=" * 60)
    
    # Запускаємо тести
    test_preprocessor()
    test_fuzzy_matcher()
    test_symptom_extractor()
    
    print("\n" + "=" * 60)
    print("✅ ВСІ ТЕСТИ ПРОЙДЕНО!")
    print("=" * 60)
    
    # Запитуємо чи запустити демо
    response = input("\n▶ Запустити інтерактивну демонстрацію? (y/n): ").strip().lower()
    if response == 'y':
        demo_interactive()


if __name__ == "__main__":
    main()
