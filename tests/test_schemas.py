"""
Тести для модуля schemas

Запуск: pytest tests/test_schemas.py -v
Або демо: python tests/test_schemas.py
"""

from datetime import datetime


def test_symptom():
    """Тест моделі Symptom"""
    from dr_case.schemas import Symptom, SymptomStatus
    
    # Базове створення
    symptom = Symptom(name="Fever", severity=0.8)
    
    assert symptom.name == "fever"  # нормалізовано в lowercase
    assert symptom.severity == 0.8
    assert symptom.status == SymptomStatus.PRESENT
    
    print(f"✓ Symptom: {symptom.name}, severity={symptom.severity}")
    
    # З негативним статусом
    absent = Symptom(name="Rash", status=SymptomStatus.ABSENT)
    assert absent.status == SymptomStatus.ABSENT
    
    print(f"✓ Absent symptom: {absent.name}")
    
    return True


def test_patient():
    """Тест моделі Patient"""
    from dr_case.schemas import Patient, Gender
    
    patient = Patient(
        patient_id="P001",
        age=35,
        gender=Gender.MALE
    )
    
    assert patient.age == 35
    assert patient.gender == Gender.MALE
    
    print(f"✓ Patient: id={patient.patient_id}, age={patient.age}, gender={patient.gender.value}")
    
    return True


def test_vitals():
    """Тест моделі Vitals"""
    from dr_case.schemas import Vitals
    
    vitals = Vitals(
        temperature_c=38.5,
        heart_rate=92,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        spo2=97.0,
        weight_kg=75.0,
        height_cm=180.0
    )
    
    assert vitals.temperature_c == 38.5
    assert vitals.blood_pressure == "120/80"
    assert vitals.bmi == 23.1  # 75 / (1.8^2)
    
    print(f"✓ Vitals: temp={vitals.temperature_c}°C, BP={vitals.blood_pressure}, BMI={vitals.bmi}")
    
    return True


def test_case_record():
    """Тест моделі CaseRecord"""
    from dr_case.schemas import CaseRecord, Patient, Symptom, Gender, SymptomStatus
    
    case = CaseRecord(
        case_id="CASE001",
        patient=Patient(age=35, gender=Gender.MALE),
        chief_complaint="Головний біль та температура вже 3 дні",
        symptoms=[
            Symptom(name="fever", severity=0.8),
            Symptom(name="headache", severity=0.7),
            Symptom(name="cough", severity=0.5),
            Symptom(name="rash", status=SymptomStatus.ABSENT),
        ]
    )
    
    assert case.case_id == "CASE001"
    assert len(case.symptoms) == 4
    assert len(case.present_symptoms) == 3
    assert len(case.absent_symptoms) == 1
    
    print(f"✓ CaseRecord: id={case.case_id}")
    print(f"  Present symptoms: {case.present_symptom_names}")
    print(f"  Absent symptoms: {case.absent_symptom_names}")
    
    # Тест add_symptom
    case.add_symptom("fatigue", severity=0.6)
    assert len(case.present_symptoms) == 4
    print(f"✓ Added symptom: fatigue")
    
    # Тест negate_symptom
    case.negate_symptom("vomiting")
    assert len(case.absent_symptoms) == 2
    print(f"✓ Negated symptom: vomiting")
    
    # Серіалізація в JSON
    json_str = case.model_dump_json(indent=2)
    assert len(json_str) > 0
    print(f"✓ JSON serialization: {len(json_str)} chars")
    
    return True


def test_diagnosis_hypothesis():
    """Тест моделі DiagnosisHypothesis"""
    from dr_case.schemas import DiagnosisHypothesis, ConfidenceLevel
    
    hypothesis = DiagnosisHypothesis(
        disease_name="Influenza",
        confidence=0.85,
        matching_symptoms=["fever", "headache", "cough", "fatigue"],
        missing_symptoms=["runny nose"],
        confidence_change=0.12
    )
    
    assert hypothesis.confidence == 0.85
    assert hypothesis.confidence_level == ConfidenceLevel.HIGH
    assert hypothesis.trend == "rising"
    
    print(f"✓ DiagnosisHypothesis: {hypothesis.disease_name}")
    print(f"  Confidence: {hypothesis.confidence} ({hypothesis.confidence_level.value})")
    print(f"  Trend: {hypothesis.trend} (change: {hypothesis.confidence_change:+.2f})")
    
    return True


def test_diagnosis_result():
    """Тест моделі DiagnosisResult"""
    from dr_case.schemas import DiagnosisResult, DiagnosisHypothesis, StopReason
    
    result = DiagnosisResult(
        case_id="CASE001",
        stop_reason=StopReason.DOMINANCE,
        top_diagnosis=DiagnosisHypothesis(
            disease_name="Influenza",
            confidence=0.87,
            matching_symptoms=["fever", "headache", "cough"]
        ),
        hypotheses=[
            DiagnosisHypothesis(disease_name="Influenza", confidence=0.87),
            DiagnosisHypothesis(disease_name="Common Cold", confidence=0.45),
            DiagnosisHypothesis(disease_name="COVID-19", confidence=0.32),
        ],
        iterations_count=5,
        questions_asked=4
    )
    
    assert result.stop_reason == StopReason.DOMINANCE
    assert result.top_confidence == 0.87
    assert result.is_confident == True
    
    print(f"✓ DiagnosisResult:")
    print(f"  Stop reason: {result.stop_reason.value}")
    print(f"  Top diagnosis: {result.top_diagnosis.disease_name} ({result.top_confidence:.0%})")
    print(f"  Iterations: {result.iterations_count}, Questions: {result.questions_asked}")
    
    # Summary
    summary = result.to_summary()
    print(f"✓ Summary: {summary}")
    
    return True


def test_iteration_state():
    """Тест моделі IterationState"""
    from dr_case.schemas import IterationState, SOMResult, Question
    
    state = IterationState(
        iteration=3,
        present_symptoms=["fever", "headache", "cough"],
        absent_symptoms=["rash"],
        som_result=SOMResult(
            bmu_coords=(3, 4),
            bmu_distance=0.23,
            memberships={"3_4": 0.45, "3_5": 0.30, "4_4": 0.25},
            top_units=["3_4", "3_5", "4_4"],
            active_units_count=3,
            cumulative_mass=1.0
        ),
        candidate_diseases=["Influenza", "Common Cold", "COVID-19"],
        disease_scores={
            "Influenza": 0.78,
            "Common Cold": 0.45,
            "COVID-19": 0.32
        },
        question=Question(
            symptom="muscle_pain",
            text="Чи відчуваєте ви біль у м'язах?",
            information_gain=0.34
        )
    )
    
    assert state.iteration == 3
    assert state.symptom_count == 4
    assert state.candidate_count == 3
    assert state.top_disease == "Influenza"
    assert state.top_score == 0.78
    
    print(f"✓ IterationState: iteration={state.iteration}")
    print(f"  Symptoms: {state.symptom_count} (present: {len(state.present_symptoms)}, absent: {len(state.absent_symptoms)})")
    print(f"  Candidates: {state.candidate_count}")
    print(f"  Top disease: {state.top_disease} ({state.top_score:.2f})")
    print(f"  BMU: {state.som_result.bmu_id} (distance: {state.som_result.bmu_distance})")
    
    return True


def test_session_state():
    """Тест моделі SessionState"""
    from dr_case.schemas import SessionState, IterationState
    
    session = SessionState(
        session_id="sess_12345",
        case_id="CASE001"
    )
    
    # Додаємо ітерації
    for i in range(3):
        state = IterationState(
            iteration=i,
            present_symptoms=["fever", "headache"],
            disease_scores={"Influenza": 0.5 + i * 0.1}
        )
        session.add_iteration(state)
    
    assert session.current_iteration == 2
    assert len(session.iterations) == 3
    assert session.latest_iteration.iteration == 2
    
    print(f"✓ SessionState: {session.session_id}")
    print(f"  Current iteration: {session.current_iteration}")
    print(f"  Total iterations: {len(session.iterations)}")
    print(f"  Duration: {session.duration_seconds:.2f}s")
    
    # Виключення діагнозу
    session.exclude_disease("Common Cold", "Ruled out by symptoms")
    assert "Common Cold" in session.excluded_diseases
    print(f"✓ Excluded: {session.excluded_diseases}")
    
    # Відновлення
    restored = session.restore_excluded()
    assert len(session.excluded_diseases) == 0
    print(f"✓ Restored: {restored}")
    
    return True


def demo():
    """Повна демонстрація модуля schemas"""
    print("=" * 60)
    print("Dr.Case — Демонстрація модуля schemas")
    print("=" * 60)
    
    try:
        print("\n--- 1. Symptom ---")
        test_symptom()
        
        print("\n--- 2. Patient ---")
        test_patient()
        
        print("\n--- 3. Vitals ---")
        test_vitals()
        
        print("\n--- 4. CaseRecord ---")
        test_case_record()
        
        print("\n--- 5. DiagnosisHypothesis ---")
        test_diagnosis_hypothesis()
        
        print("\n--- 6. DiagnosisResult ---")
        test_diagnosis_result()
        
        print("\n--- 7. IterationState ---")
        test_iteration_state()
        
        print("\n--- 8. SessionState ---")
        test_session_state()
        
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
