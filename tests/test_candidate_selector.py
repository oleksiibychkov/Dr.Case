"""
Тести для модуля candidate_selector

Запуск: pytest tests/test_candidate_selector.py -v
Або демо: python tests/test_candidate_selector.py
"""

from pathlib import Path

# Шляхи
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "som_optimized.pkl"


def test_selector_from_model():
    """Тест створення з файлу моделі"""
    from dr_case.candidate_selector import CandidateSelector
    
    if not MODEL_PATH.exists():
        print(f"⚠ Model not found: {MODEL_PATH}")
        print("  Run optimization first or use test_selector_from_database()")
        return None
    
    selector = CandidateSelector.from_model_file(
        model_path=str(MODEL_PATH),
        database_path=str(DATA_PATH)
    )
    
    print(f"✓ CandidateSelector loaded: {selector}")
    print(f"  Grid: {selector.grid_shape}")
    print(f"  Symptoms: {selector.n_symptoms}")
    
    return selector


def test_selector_from_database():
    """Тест створення та навчання з бази даних"""
    from dr_case.candidate_selector import CandidateSelector
    
    if not DATA_PATH.exists():
        print(f"⚠ Database not found: {DATA_PATH}")
        return None
    
    print("Training SOM from database (this may take a moment)...")
    
    selector = CandidateSelector.from_database(
        database_path=str(DATA_PATH),
        som_config={"grid_height": 15, "grid_width": 15},
        epochs=100  # Менше для швидкості тесту
    )
    
    print(f"✓ CandidateSelector created: {selector}")
    
    return selector


def test_basic_selection():
    """Тест базового відбору"""
    selector = _get_selector()
    if selector is None:
        return None
    
    # Симптоми грипу
    symptoms = ["fever", "headache", "cough", "fatigue"]
    
    result = selector.select(symptoms)
    
    assert result.candidate_count > 0
    assert len(result.candidates) == result.candidate_count
    assert result.bmu is not None
    
    print(f"\n✓ Basic selection test")
    print(f"  Input symptoms: {symptoms}")
    print(f"  BMU: {result.bmu}")
    print(f"  Active units: {len(result.active_units)}")
    print(f"  Candidates: {result.candidate_count}")
    print(f"  Top 5 candidates:")
    for i, disease in enumerate(result.candidates[:5]):
        print(f"    {i+1}. {disease}")
    
    return result


def test_selection_with_negatives():
    """Тест відбору з негативними симптомами"""
    selector = _get_selector()
    if selector is None:
        return None
    
    present = ["fever", "cough", "shortness of breath"]
    absent = ["rash", "vomiting", "diarrhea"]
    
    result = selector.select(
        present_symptoms=present,
        absent_symptoms=absent
    )
    
    assert result.candidate_count > 0
    
    print(f"\n✓ Selection with negatives test")
    print(f"  Present: {present}")
    print(f"  Absent: {absent}")
    print(f"  Candidates: {result.candidate_count}")
    print(f"  Top 5:")
    for i, disease in enumerate(result.candidates[:5]):
        print(f"    {i+1}. {disease}")
    
    return result


def test_selection_from_case():
    """Тест відбору з CaseRecord"""
    from dr_case.schemas import CaseRecord, Patient, Symptom, SymptomStatus, Gender
    
    selector = _get_selector()
    if selector is None:
        return None
    
    # Створюємо випадок
    case = CaseRecord(
        case_id="TEST001",
        patient=Patient(age=45, gender=Gender.MALE),
        chief_complaint="Кашель та температура протягом 3 днів",
        symptoms=[
            Symptom(name="fever", severity=0.8),
            Symptom(name="cough", severity=0.7),
            Symptom(name="fatigue", severity=0.6),
            Symptom(name="rash", status=SymptomStatus.ABSENT),
        ]
    )
    
    result = selector.select_from_case(case)
    
    assert result.candidate_count > 0
    assert "fever" in result.present_symptoms
    assert "rash" in result.absent_symptoms
    
    print(f"\n✓ Selection from CaseRecord test")
    print(f"  Case: {case.case_id}")
    print(f"  Present symptoms: {result.present_symptoms}")
    print(f"  Absent symptoms: {result.absent_symptoms}")
    print(f"  Candidates: {result.candidate_count}")
    
    return result


def test_conversion_to_schemas():
    """Тест конвертації в schemas"""
    selector = _get_selector()
    if selector is None:
        return None
    
    result = selector.select(["fever", "headache"])
    
    # Конвертуємо в CandidateDiagnoses
    candidates_schema = result.to_candidate_diagnoses()
    
    assert len(candidates_schema.candidates) == result.candidate_count
    assert candidates_schema.bmu_id is not None
    
    print(f"\n✓ CandidateDiagnoses schema:")
    print(f"  Candidates: {len(candidates_schema.candidates)}")
    print(f"  BMU: {candidates_schema.bmu_id}")
    
    # Конвертуємо в SOMResult
    som_result = result.to_som_result()
    
    assert som_result.bmu_coords == result.bmu
    assert som_result.active_units_count == len(result.active_units)
    
    print(f"\n✓ SOMResult schema:")
    print(f"  BMU coords: {som_result.bmu_coords}")
    print(f"  Active units: {som_result.active_units_count}")
    
    return True


def test_neighborhood_analysis():
    """Тест аналізу сусідства BMU"""
    selector = _get_selector()
    if selector is None:
        return None
    
    symptoms = ["chest tightness", "shortness of breath", "anxiety and nervousness"]
    
    neighborhood = selector.get_diseases_in_neighborhood(symptoms, radius=1)
    
    print(f"\n✓ Neighborhood analysis test")
    print(f"  Symptoms: {symptoms}")
    print(f"  Units with diseases: {len(neighborhood)}")
    
    for unit_id, diseases in list(neighborhood.items())[:3]:
        print(f"  Unit {unit_id}: {len(diseases)} diseases")
        for d in diseases[:3]:
            print(f"    - {d}")
    
    return neighborhood


def test_config_update():
    """Тест оновлення конфігурації"""
    selector = _get_selector()
    if selector is None:
        return None
    
    # Початкові результати
    result1 = selector.select(["fever", "headache"])
    count1 = result1.candidate_count
    
    # Оновлюємо параметри (більше кандидатів)
    selector.update_config(alpha=0.99, k=15)
    
    result2 = selector.select(["fever", "headache"])
    count2 = result2.candidate_count
    
    print(f"\n✓ Config update test")
    print(f"  Before (α=0.9, k=6): {count1} candidates")
    print(f"  After (α=0.99, k=15): {count2} candidates")
    
    # Повертаємо назад
    selector.update_config(alpha=0.9, k=6)
    
    return True


def test_batch_selection():
    """Тест batch відбору"""
    selector = _get_selector()
    if selector is None:
        return None
    
    symptom_lists = [
        ["fever", "headache"],
        ["cough", "shortness of breath"],
        ["nausea", "vomiting", "abdominal_pain"],
        ["rash", "itching"],
    ]
    
    results = selector.select_batch(symptom_lists)
    
    assert len(results) == len(symptom_lists)
    
    print(f"\n✓ Batch selection test")
    for i, (symptoms, result) in enumerate(zip(symptom_lists, results)):
        print(f"  {i+1}. {symptoms[:2]}... → {result.candidate_count} candidates")
    
    return results


def _get_selector():
    """Отримати selector (з моделі або навчити)"""
    if MODEL_PATH.exists():
        return test_selector_from_model()
    else:
        return test_selector_from_database()


def demo():
    """Повна демонстрація модуля candidate_selector"""
    print("=" * 60)
    print("Dr.Case — Демонстрація Candidate Selector")
    print("=" * 60)
    
    if not DATA_PATH.exists():
        print(f"\n❌ ПОМИЛКА: База даних не знайдена!")
        print(f"   Очікуваний шлях: {DATA_PATH}")
        return False
    
    try:
        print("\n--- 1. Loading/Creating Selector ---")
        selector = _get_selector()
        
        if selector is None:
            return False
        
        print("\n--- 2. Basic Selection ---")
        test_basic_selection()
        
        print("\n--- 3. Selection with Negatives ---")
        test_selection_with_negatives()
        
        print("\n--- 4. Selection from CaseRecord ---")
        test_selection_from_case()
        
        print("\n--- 5. Schema Conversion ---")
        test_conversion_to_schemas()
        
        print("\n--- 6. Neighborhood Analysis ---")
        test_neighborhood_analysis()
        
        print("\n--- 7. Config Update ---")
        test_config_update()
        
        print("\n--- 8. Batch Selection ---")
        test_batch_selection()
        
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
