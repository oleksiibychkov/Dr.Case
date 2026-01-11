"""
Тести для модуля api

Запуск: pytest tests/test_api.py -v
Або демо: python tests/test_api.py
"""

from pathlib import Path
import sys

# Шляхи
DATA_PATH = Path(__file__).parent.parent / "data" / "unified_disease_symptom_data_full.json"
SOM_MODEL_PATH = Path(__file__).parent.parent / "models" / "som_optimized.pkl"
NN_MODEL_PATH = Path(__file__).parent.parent / "models" / "nn_model.pt"


def test_imports():
    """Тест імпортів"""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        print("✓ FastAPI imports OK")
        return True
    except ImportError as e:
        print(f"❌ FastAPI not installed: {e}")
        print("   Run: pip install fastapi uvicorn")
        return False


def test_app_creation():
    """Тест створення FastAPI app"""
    from dr_case.api.main import app
    from fastapi import FastAPI
    
    assert isinstance(app, FastAPI)
    print(f"✓ FastAPI app created: {app.title}")
    
    return app


def test_client_setup():
    """Тест налаштування test client"""
    from fastapi.testclient import TestClient
    from dr_case.api.main import app
    from dr_case.api.dependencies import app_state
    
    if not DATA_PATH.exists() or not SOM_MODEL_PATH.exists():
        print(f"⚠ Skipping: required files not found")
        return None
    
    # Ініціалізуємо
    app_state.initialize(
        som_model_path=str(SOM_MODEL_PATH),
        nn_model_path=str(NN_MODEL_PATH) if NN_MODEL_PATH.exists() else None,
        database_path=str(DATA_PATH)
    )
    
    client = TestClient(app)
    print("✓ Test client ready")
    
    return client


def test_root_endpoint():
    """Тест кореневого endpoint"""
    client = _get_client()
    if client is None:
        return None
    
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Dr.Case API"
    
    print(f"✓ Root endpoint: {data}")
    
    return data


def test_health_endpoint():
    """Тест health check"""
    client = _get_client()
    if client is None:
        return None
    
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    
    print(f"✓ Health check:")
    print(f"  Status: {data['status']}")
    print(f"  Components: {data['components']}")
    
    return data


def test_quick_diagnosis():
    """Тест швидкої діагностики"""
    client = _get_client()
    if client is None:
        return None
    
    response = client.post(
        "/api/v1/diagnose/quick",
        json={
            "present_symptoms": ["fever", "cough", "headache"],
            "absent_symptoms": ["rash"]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "hypotheses" in data
    assert len(data["hypotheses"]) > 0
    assert data["top_diagnosis"] is not None
    
    print(f"\n✓ Quick diagnosis:")
    print(f"  Symptoms: fever, cough, headache")
    print(f"  Candidates: {data['candidates_count']}")
    print(f"  Top diagnosis: {data['top_diagnosis']}")
    print(f"  Confidence: {data['top_confidence']:.1%}")
    print(f"  Top 3:")
    
    for h in data["hypotheses"][:3]:
        print(f"    - {h['disease_name']}: {h['confidence_percent']}")
    
    return data


def test_start_session():
    """Тест початку сесії"""
    client = _get_client()
    if client is None:
        return None
    
    response = client.post(
        "/api/v1/session/start",
        json={
            "initial_symptoms": ["fever", "cough"],
            "patient_age": 35,
            "patient_gender": "male"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "session_id" in data
    assert data["status"] == "active"
    assert len(data["next_questions"]) > 0
    
    print(f"\n✓ Session started:")
    print(f"  Session ID: {data['session_id']}")
    print(f"  Candidates: {data['candidates_count']}")
    print(f"  Top diagnosis: {data['top_diagnosis']}")
    print(f"  Questions:")
    
    for q in data["next_questions"][:3]:
        print(f"    - {q['text']} (IG={q['information_gain']:.3f})")
    
    return data


def test_answer_question():
    """Тест відповіді на питання"""
    client = _get_client()
    if client is None:
        return None
    
    # Починаємо сесію
    start_response = client.post(
        "/api/v1/session/start",
        json={"initial_symptoms": ["fever", "headache"]}
    )
    session_id = start_response.json()["session_id"]
    
    # Відповідаємо на питання
    first_question = start_response.json()["next_questions"][0]
    
    response = client.post(
        f"/api/v1/session/{session_id}/answer",
        json={
            "symptom": first_question["symptom"],
            "answer": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"\n✓ Answer processed:")
    print(f"  Question: {first_question['symptom']}?")
    print(f"  Answer: Yes")
    print(f"  New top: {data['top_diagnosis']} ({data['top_confidence']:.1%})")
    print(f"  Candidates: {data['candidates_count']}")
    
    return data


def test_complete_session():
    """Тест завершення сесії"""
    client = _get_client()
    if client is None:
        return None
    
    # Починаємо сесію
    start_response = client.post(
        "/api/v1/session/start",
        json={"initial_symptoms": ["fever", "cough", "fatigue"]}
    )
    session_id = start_response.json()["session_id"]
    
    # Відповідаємо на кілька питань
    for _ in range(2):
        questions = start_response.json().get("next_questions", [])
        if questions:
            client.post(
                f"/api/v1/session/{session_id}/answer",
                json={"symptom": questions[0]["symptom"], "answer": True}
            )
    
    # Завершуємо
    response = client.post(f"/api/v1/session/{session_id}/complete")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "completed"
    assert "explanation" in data
    
    print(f"\n✓ Session completed:")
    print(f"  Questions asked: {data['questions_asked']}")
    print(f"  Top diagnosis: {data['top_diagnosis']}")
    print(f"  Confidence: {data['confidence_level']}")
    print(f"\n  Explanation (first 500 chars):")
    print(f"  {data['explanation'][:500]}...")
    
    return data


def test_list_symptoms():
    """Тест списку симптомів"""
    client = _get_client()
    if client is None:
        return None
    
    response = client.get("/api/v1/info/symptoms")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["count"] == 461
    
    print(f"\n✓ Symptoms list:")
    print(f"  Count: {data['count']}")
    print(f"  First 5: {data['symptoms'][:5]}")
    
    return data


def test_search_symptoms():
    """Тест пошуку симптомів"""
    client = _get_client()
    if client is None:
        return None
    
    response = client.get("/api/v1/info/symptoms/search?q=fever")
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"\n✓ Search 'fever':")
    print(f"  Found: {data['count']}")
    print(f"  Results: {data['symptoms']}")
    
    return data


def test_full_interactive_flow():
    """Тест повного інтерактивного потоку"""
    client = _get_client()
    if client is None:
        return None
    
    print(f"\n✓ Full interactive flow:")
    
    # 1. Пошук симптомів
    print(f"\n  1. Searching symptoms...")
    search = client.get("/api/v1/info/symptoms/search?q=cough").json()
    print(f"     Found {search['count']} symptoms with 'cough'")
    
    # 2. Початок сесії
    print(f"\n  2. Starting session with ['fever', 'cough']...")
    start = client.post(
        "/api/v1/session/start",
        json={"initial_symptoms": ["fever", "cough"]}
    ).json()
    
    session_id = start["session_id"]
    print(f"     Session: {session_id}")
    print(f"     Initial top: {start['top_diagnosis']} ({start['top_confidence']:.1%})")
    
    # 3. Відповідаємо на питання
    print(f"\n  3. Answering questions...")
    
    current = start
    for i in range(3):
        if not current.get("next_questions"):
            break
        
        q = current["next_questions"][0]
        # Симулюємо відповідь (чергуємо yes/no)
        answer = i % 2 == 0
        
        current = client.post(
            f"/api/v1/session/{session_id}/answer",
            json={"symptom": q["symptom"], "answer": answer}
        ).json()
        
        answer_str = "Yes" if answer else "No"
        print(f"     Q{i+1}: {q['symptom']}? → {answer_str}")
        print(f"         Top: {current['top_diagnosis']} ({current['top_confidence']:.1%})")
    
    # 4. Завершення
    print(f"\n  4. Completing session...")
    result = client.post(f"/api/v1/session/{session_id}/complete").json()
    
    print(f"     Final diagnosis: {result['top_diagnosis']}")
    print(f"     Confidence: {result['confidence_level']}")
    print(f"     Questions asked: {result['questions_asked']}")
    
    return result


# === Client Cache ===

_client = None

def _get_client():
    """Отримати або створити test client"""
    global _client
    
    if _client is None:
        _client = test_client_setup()
    
    return _client


def demo():
    """Повна демонстрація API"""
    print("=" * 60)
    print("Dr.Case — Демонстрація REST API")
    print("=" * 60)
    
    # Перевірка FastAPI
    if not test_imports():
        print("\n❌ Встановіть FastAPI: pip install fastapi uvicorn")
        return False
    
    if not DATA_PATH.exists() or not SOM_MODEL_PATH.exists():
        print(f"\n❌ ПОМИЛКА: Необхідні файли не знайдені!")
        print(f"   Database: {DATA_PATH.exists()}")
        print(f"   SOM Model: {SOM_MODEL_PATH.exists()}")
        return False
    
    try:
        print("\n--- 1. App Creation ---")
        test_app_creation()
        
        print("\n--- 2. Client Setup ---")
        test_client_setup()
        
        print("\n--- 3. Root Endpoint ---")
        test_root_endpoint()
        
        print("\n--- 4. Health Check ---")
        test_health_endpoint()
        
        print("\n--- 5. Quick Diagnosis ---")
        test_quick_diagnosis()
        
        print("\n--- 6. Start Session ---")
        test_start_session()
        
        print("\n--- 7. Answer Question ---")
        test_answer_question()
        
        print("\n--- 8. Complete Session ---")
        test_complete_session()
        
        print("\n--- 9. List Symptoms ---")
        test_list_symptoms()
        
        print("\n--- 10. Search Symptoms ---")
        test_search_symptoms()
        
        print("\n--- 11. Full Interactive Flow ---")
        test_full_interactive_flow()
        
        print("\n" + "=" * 60)
        print("✅ Всі тести пройдено успішно!")
        print("=" * 60)
        
        print("\n📋 Для запуску сервера:")
        print("   cd C:\\Projects\\Dr.Case")
        print("   uvicorn dr_case.api.main:app --reload --port 8000")
        print("\n   Swagger UI: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo()
