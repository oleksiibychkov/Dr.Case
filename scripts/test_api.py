#!/usr/bin/env python3
"""
Dr.Case — Тестування API

Запуск:
    1. Спочатку запусти сервер:
       python scripts/run_api.py
    
    2. В іншому терміналі запусти тести:
       python scripts/test_api.py

Або з curl:
    curl http://localhost:8000/health
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def print_result(success, message):
    icon = "✅" if success else "❌"
    print(f"   {icon} {message}")

def test_health():
    """Тест 1: Health check"""
    print_header("TEST 1: Health Check")
    
    try:
        r = requests.get(f"{BASE_URL}/health")
        data = r.json()
        
        print_result(r.status_code == 200, f"Status: {r.status_code}")
        print_result(data.get("status") == "ok", f"API Status: {data.get('status')}")
        print_result(data.get("models_loaded"), f"Models loaded: {data.get('models_loaded')}")
        print(f"   📊 Symptoms: {data.get('database_symptoms')}")
        print(f"   📊 Diseases: {data.get('database_diseases')}")
        
        return r.status_code == 200
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_symptoms_list():
    """Тест 2: Список симптомів"""
    print_header("TEST 2: List Symptoms")
    
    try:
        r = requests.get(f"{BASE_URL}/api/symptoms?limit=10")
        data = r.json()
        
        print_result(r.status_code == 200, f"Status: {r.status_code}")
        print_result(len(data) > 0, f"Symptoms returned: {len(data)}")
        
        print("   📋 First 5 symptoms:")
        for s in data[:5]:
            print(f"      - {s['name']}")
        
        return r.status_code == 200
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_symptoms_search():
    """Тест 3: Пошук симптомів"""
    print_header("TEST 3: Search Symptoms")
    
    try:
        r = requests.get(f"{BASE_URL}/api/symptoms/search?q=head")
        data = r.json()
        
        print_result(r.status_code == 200, f"Status: {r.status_code}")
        print_result(data.get("total", 0) > 0, f"Results found: {data.get('total')}")
        
        print(f"   🔍 Query: '{data.get('query')}'")
        print("   📋 Results:")
        for s in data.get("results", [])[:5]:
            print(f"      - {s['name']}")
        
        return r.status_code == 200
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_nlp_extract():
    """Тест 4: NLP витягування симптомів"""
    print_header("TEST 4: NLP Extract Symptoms")
    
    test_cases = [
        "Болить голова і температура 38.5",
        "I have a cough and sore throat for 3 days",
        "Нудота, блювота, біль у животі",
    ]
    
    success = True
    
    for text in test_cases:
        try:
            r = requests.post(
                f"{BASE_URL}/api/symptoms/extract",
                json={"text": text}
            )
            data = r.json()
            
            found = len(data.get("symptoms", []))
            print_result(found > 0, f"'{text[:30]}...' → {found} symptoms")
            
            if data.get("symptoms"):
                print(f"      Symptoms: {data['symptoms']}")
            if data.get("vitals"):
                print(f"      Vitals: {data['vitals']}")
            if data.get("duration"):
                print(f"      Duration: {data['duration']}")
            
            if found == 0:
                success = False
                
        except Exception as e:
            print_result(False, f"Error: {e}")
            success = False
    
    return success

def test_quick_diagnose():
    """Тест 5: Швидка діагностика"""
    print_header("TEST 5: Quick Diagnose")
    
    try:
        r = requests.post(
            f"{BASE_URL}/api/diagnose",
            json={
                "symptoms": ["Headache", "Fever", "Cough"],
                "top_k": 5
            }
        )
        data = r.json()
        
        print_result(r.status_code == 200, f"Status: {r.status_code}")
        
        hypotheses = data.get("hypotheses", [])
        print_result(len(hypotheses) > 0, f"Hypotheses: {len(hypotheses)}")
        print(f"   ⏱️ Processing time: {data.get('processing_time_ms', 0):.1f}ms")
        
        print("   🎯 Top diagnoses:")
        for h in hypotheses[:5]:
            prob = h.get("probability", 0) * 100
            print(f"      {h.get('rank')}. {h.get('disease')} ({prob:.1f}%)")
        
        return r.status_code == 200 and len(hypotheses) > 0
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_diagnose_from_text():
    """Тест 6: Діагностика з тексту"""
    print_header("TEST 6: Diagnose from Text")
    
    try:
        text = "Болить голова, температура 38, кашель"
        r = requests.post(
            f"{BASE_URL}/api/diagnose/text?text={text}&top_k=5"
        )
        data = r.json()
        
        print_result(r.status_code == 200, f"Status: {r.status_code}")
        print(f"   📝 Input: '{text}'")
        print(f"   📋 Extracted symptoms: {data.get('symptoms')}")
        
        hypotheses = data.get("hypotheses", [])
        print("   🎯 Top diagnoses:")
        for h in hypotheses[:3]:
            prob = h.get("probability", 0) * 100
            print(f"      {h.get('rank')}. {h.get('disease')} ({prob:.1f}%)")
        
        return r.status_code == 200
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_session_workflow():
    """Тест 7: Повний workflow сесії"""
    print_header("TEST 7: Session Workflow")
    
    session_id = None
    
    try:
        # 7.1 Створення сесії
        print("\n   📌 Step 1: Create session")
        r = requests.post(
            f"{BASE_URL}/api/sessions",
            json={
                "symptoms": ["Headache", "Fever"],
                "language": "uk"
            }
        )
        data = r.json()
        
        session_id = data.get("session_id")
        print_result(session_id is not None, f"Session created: {session_id}")
        print(f"      Status: {data.get('status')}")
        print(f"      Iteration: {data.get('iteration')}")
        
        if data.get("hypotheses"):
            print(f"      Top hypothesis: {data['hypotheses'][0]['disease']}")
        
        # 7.2 Отримання стану
        print("\n   📌 Step 2: Get session state")
        r = requests.get(f"{BASE_URL}/api/sessions/{session_id}")
        data = r.json()
        
        print_result(r.status_code == 200, f"Session retrieved")
        print(f"      Confirmed symptoms: {data.get('confirmed_symptoms')}")
        
        if data.get("current_question"):
            q = data["current_question"]
            print(f"      Question: {q.get('text_uk')}")
        
        # 7.3 Відповідь на питання (якщо є)
        if data.get("current_question") and data.get("status") == "waiting_answer":
            print("\n   📌 Step 3: Answer question")
            r = requests.post(
                f"{BASE_URL}/api/sessions/{session_id}/answer",
                json={"answer": True}
            )
            data = r.json()
            
            print_result(data.get("accepted"), "Answer accepted")
            state = data.get("session_state", {})
            print(f"      New iteration: {state.get('iteration')}")
        
        # 7.4 Feedback
        print("\n   📌 Step 4: Submit feedback")
        r = requests.post(
            f"{BASE_URL}/api/sessions/{session_id}/feedback",
            json={
                "feedback_type": "treatment_success",
                "comment": "Test feedback"
            }
        )
        data = r.json()
        
        print_result(data.get("accepted"), f"Feedback: {data.get('action_taken')}")
        
        # 7.5 Видалення сесії
        print("\n   📌 Step 5: Delete session")
        r = requests.delete(f"{BASE_URL}/api/sessions/{session_id}")
        data = r.json()
        
        print_result(data.get("deleted"), "Session deleted")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        
        # Cleanup
        if session_id:
            try:
                requests.delete(f"{BASE_URL}/api/sessions/{session_id}")
            except:
                pass
        
        return False

def test_diseases():
    """Тест 8: Список хвороб"""
    print_header("TEST 8: List Diseases")
    
    try:
        r = requests.get(f"{BASE_URL}/api/diseases?limit=10")
        data = r.json()
        
        print_result(r.status_code == 200, f"Status: {r.status_code}")
        print(f"   📊 Total diseases: {data.get('total')}")
        
        print("   📋 First 5 diseases:")
        for d in data.get("diseases", [])[:5]:
            print(f"      - {d}")
        
        return r.status_code == 200
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🏥 Dr.Case — API Test Suite")
    print("=" * 60)
    print(f"   Target: {BASE_URL}")
    print("=" * 60)
    
    # Перевіряємо чи сервер доступний
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print("\n❌ ERROR: Server not available!")
        print("   Make sure to run: python scripts/run_api.py")
        return
    
    # Запускаємо тести
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("List Symptoms", test_symptoms_list()))
    results.append(("Search Symptoms", test_symptoms_search()))
    results.append(("NLP Extract", test_nlp_extract()))
    results.append(("Quick Diagnose", test_quick_diagnose()))
    results.append(("Diagnose from Text", test_diagnose_from_text()))
    results.append(("Session Workflow", test_session_workflow()))
    results.append(("List Diseases", test_diseases()))
    
    # Підсумок
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"   {icon} {name}")
    
    print("=" * 60)
    print(f"   Total: {passed}/{total} passed")
    
    if passed == total:
        print("   🎉 All tests passed!")
    else:
        print(f"   ⚠️ {total - passed} test(s) failed")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
