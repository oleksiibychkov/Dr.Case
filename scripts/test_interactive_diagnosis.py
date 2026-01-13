"""
Тест інтерактивної діагностики Dr.Case

Перевіряє:
1. Завантаження движка
2. Швидка діагностика (без питань)
3. Генерація питань
4. Обробка відповідей
5. Оновлення гіпотез після відповідей
"""

import sys
from pathlib import Path

# Додаємо шлях до проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_hypotheses(hypotheses, title="Гіпотези"):
    print(f"\n{title}:")
    for i, h in enumerate(hypotheses[:5], 1):
        if hasattr(h, 'disease'):
            print(f"  {i}. {h.disease}: {h.probability:.2%}")
        elif hasattr(h, 'disease_name'):
            print(f"  {i}. {h.disease_name}: {h.confidence:.2%}")
        elif isinstance(h, tuple):
            print(f"  {i}. {h[0]}: {h[1]:.2%}")
        elif isinstance(h, dict):
            print(f"  {i}. {h.get('disease', h.get('name', 'Unknown'))}: {h.get('probability', h.get('confidence', 0)):.2%}")

def test_house_flow_engine():
    """Тест HouseFlowEngine (основний движок)"""
    print_header("Тест HouseFlowEngine")
    
    try:
        from dr_case.diagnosis_engine import HouseFlowEngine
        print("✓ HouseFlowEngine імпортовано")
    except ImportError as e:
        print(f"✗ Помилка імпорту HouseFlowEngine: {e}")
        return False
    
    # Ініціалізація
    try:
        engine = HouseFlowEngine()
        print(f"✓ HouseFlowEngine створено")
        print(f"  - Хвороб: {len(engine.database) if hasattr(engine, 'database') else 'N/A'}")
    except Exception as e:
        print(f"✗ Помилка створення HouseFlowEngine: {e}")
        return False
    
    # Тест 1: Швидка діагностика
    print("\n--- Тест 1: Швидка діагностика ---")
    symptoms = ["Fever", "Cough", "Headache"]
    print(f"Симптоми: {symptoms}")
    
    try:
        results = engine.quick_diagnosis(symptoms)
        print(f"✓ quick_diagnosis працює, отримано {len(results)} результатів")
        print_hypotheses(results)
    except Exception as e:
        print(f"✗ Помилка quick_diagnosis: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 2: Створення сесії
    print("\n--- Тест 2: Створення інтерактивної сесії ---")
    try:
        session = engine.start_session(symptoms)
        print(f"✓ Сесія створена: {session.session_id if hasattr(session, 'session_id') else 'OK'}")
        print(f"  - Статус: {session.status if hasattr(session, 'status') else 'N/A'}")
        print(f"  - Ітерація: {session.iteration if hasattr(session, 'iteration') else 'N/A'}")
        
        if hasattr(session, 'hypotheses'):
            print_hypotheses(session.hypotheses, "Початкові гіпотези")
    except Exception as e:
        print(f"✗ Помилка створення сесії: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Тест 3: Генерація питання
    print("\n--- Тест 3: Генерація питання ---")
    try:
        question = engine.get_next_question(session)
        if question:
            print(f"✓ Питання згенеровано:")
            if hasattr(question, 'symptom'):
                print(f"  - Симптом: {question.symptom}")
            if hasattr(question, 'text'):
                print(f"  - Текст: {question.text}")
            if hasattr(question, 'eig'):
                print(f"  - EIG: {question.eig:.4f}")
        else:
            print("⚠ Питання = None (можливо вже достатньо впевненості)")
    except AttributeError:
        print("⚠ Метод get_next_question не існує, пробуємо альтернативи...")
        
        # Спробувати інші методи
        for method_name in ['generate_question', 'next_question', 'ask_question']:
            if hasattr(engine, method_name):
                print(f"  Знайдено метод: {method_name}")
    except Exception as e:
        print(f"✗ Помилка генерації питання: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 4: Обробка відповіді
    print("\n--- Тест 4: Обробка відповіді ---")
    try:
        if question:
            # Симулюємо відповідь "Так"
            print(f"  Відповідаємо 'Так' на питання про {question.symptom if hasattr(question, 'symptom') else question}")
            
            result = engine.process_answer(session, True)
            print(f"✓ Відповідь оброблено")
            
            if hasattr(result, 'hypotheses'):
                print_hypotheses(result.hypotheses, "Оновлені гіпотези")
            elif hasattr(session, 'hypotheses'):
                print_hypotheses(session.hypotheses, "Оновлені гіпотези")
        else:
            print("⚠ Немає питання для тестування відповіді")
    except Exception as e:
        print(f"✗ Помилка обробки відповіді: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 5: Цикл питань
    print("\n--- Тест 5: Цикл питань (макс 3) ---")
    try:
        for i in range(3):
            question = engine.get_next_question(session)
            if not question:
                print(f"  Ітерація {i+1}: Питань більше немає")
                break
            
            symptom = question.symptom if hasattr(question, 'symptom') else str(question)
            print(f"  Ітерація {i+1}: {symptom}")
            
            # Чергуємо відповіді
            answer = (i % 2 == 0)  # True, False, True
            engine.process_answer(session, answer)
            print(f"    Відповідь: {'Так' if answer else 'Ні'}")
            
            if hasattr(session, 'hypotheses') and session.hypotheses:
                top = session.hypotheses[0]
                if hasattr(top, 'disease'):
                    print(f"    Топ-1: {top.disease} ({top.probability:.2%})")
                elif hasattr(top, 'disease_name'):
                    print(f"    Топ-1: {top.disease_name} ({top.confidence:.2%})")
    except Exception as e:
        print(f"✗ Помилка в циклі питань: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def test_diagnosis_engine():
    """Тест DiagnosisEngine (старий движок)"""
    print_header("Тест DiagnosisEngine")
    
    try:
        from dr_case.diagnosis_engine import DiagnosisEngine
        print("✓ DiagnosisEngine імпортовано")
    except ImportError as e:
        print(f"✗ Помилка імпорту DiagnosisEngine: {e}")
        return False
    
    # Перевіряємо доступні методи
    print("\nДоступні методи DiagnosisEngine:")
    for attr in dir(DiagnosisEngine):
        if not attr.startswith('_') and callable(getattr(DiagnosisEngine, attr, None)):
            print(f"  - {attr}")
    
    return True


def test_question_selector():
    """Тест QuestionSelector"""
    print_header("Тест QuestionSelector")
    
    try:
        from dr_case.question_engine import QuestionSelector
        print("✓ QuestionSelector імпортовано")
    except ImportError as e:
        print(f"✗ Помилка імпорту QuestionSelector: {e}")
        return False
    
    # Створення з бази даних
    try:
        db_path = "data/unified_disease_symptom_merged.json"
        if not Path(db_path).exists():
            db_path = "data/unified_disease_symptom_data_full.json"
        
        selector = QuestionSelector.from_database(db_path)
        print(f"✓ QuestionSelector створено з {db_path}")
    except Exception as e:
        print(f"✗ Помилка створення QuestionSelector: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Тест вибору питання
    try:
        # Імітуємо ймовірності від NN
        probs = {
            "Influenza": 0.45,
            "Common Cold": 0.30,
            "COVID-19": 0.15,
            "Bronchitis": 0.10
        }
        known = {"Fever", "Cough"}
        
        question = selector.select_question(probs, known_symptoms=known)
        
        if question:
            print(f"✓ Питання згенеровано:")
            print(f"  - Симптом: {question.symptom}")
            print(f"  - Текст: {question.text}")
            print(f"  - EIG: {question.eig:.4f}")
        else:
            print("⚠ Питання = None")
    except Exception as e:
        print(f"✗ Помилка вибору питання: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def test_api_interactive():
    """Тест через API endpoints"""
    print_header("Тест API (інтерактивна сесія)")
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Перевірка чи API працює
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            print(f"✓ API доступний: {r.status_code}")
        except:
            print("⚠ API не запущено. Запустіть: python -m uvicorn dr_case.api.app:app --port 8000")
            return False
        
        # Створення сесії
        print("\n--- Створення сесії ---")
        r = requests.post(f"{base_url}/api/sessions", json={
            "symptoms": ["Fever", "Cough", "Headache"]
        })
        
        if r.status_code == 200:
            data = r.json()
            session_id = data.get("session_id")
            print(f"✓ Сесія створена: {session_id}")
            print(f"  Гіпотези: {data.get('hypotheses', [])[:3]}")
            
            # Отримання питання
            print("\n--- Отримання питання ---")
            r = requests.get(f"{base_url}/api/sessions/{session_id}/question")
            if r.status_code == 200:
                q_data = r.json()
                print(f"✓ Питання: {q_data}")
                
                # Відповідь на питання
                if q_data.get("symptom"):
                    print("\n--- Відповідь на питання ---")
                    r = requests.post(f"{base_url}/api/sessions/{session_id}/answer", json={
                        "answer": True
                    })
                    if r.status_code == 200:
                        print(f"✓ Відповідь оброблено: {r.json()}")
            else:
                print(f"⚠ Помилка отримання питання: {r.status_code} {r.text}")
        else:
            print(f"✗ Помилка створення сесії: {r.status_code} {r.text}")
            
    except ImportError:
        print("⚠ requests не встановлено")
    except Exception as e:
        print(f"✗ Помилка: {e}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print(" Dr.Case — Тестування інтерактивної діагностики")
    print("=" * 60)
    
    # Тест 1: QuestionSelector
    test_question_selector()
    
    # Тест 2: HouseFlowEngine
    test_house_flow_engine()
    
    # Тест 3: DiagnosisEngine
    test_diagnosis_engine()
    
    # Тест 4: API (якщо запущено)
    # test_api_interactive()
    
    print("\n" + "=" * 60)
    print(" Тестування завершено")
    print("=" * 60)


if __name__ == "__main__":
    main()
