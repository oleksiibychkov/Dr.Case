"""Детальна діагностика циклу питань"""

from dr_case.diagnosis_cycle import DiagnosisCycleController, StoppingConfig

config = StoppingConfig(
    dominance_threshold=0.90,
    confidence_threshold=0.80,
    need_test_threshold=0.001,
    max_iterations=10,
    max_questions=20
)

print("Завантаження...")
controller = DiagnosisCycleController.from_models(
    database_path='data/unified_disease_symptom_merged.json',
    som_path='models/som_merged.pkl',
    nn_path='models/nn_two_branch.pt',
    stopping_config=config
)

print("\nСтарт сесії...")
result = controller.start_session(['Fever', 'Cough', 'Headache'])

print(f"\n=== Початковий стан ===")
print(f"Phase: {controller.phase if hasattr(controller, 'phase') else 'N/A'}")
print(f"Iteration: {controller.iteration if hasattr(controller, 'iteration') else 'N/A'}")
print(f"Questions asked: {controller.questions_asked if hasattr(controller, 'questions_asked') else 'N/A'}")

# Перевіряємо внутрішній стан
if hasattr(controller, 'session'):
    session = controller.session
    print(f"\nSession state:")
    print(f"  present_symptoms: {session.present_symptoms if hasattr(session, 'present_symptoms') else 'N/A'}")
    print(f"  absent_symptoms: {session.absent_symptoms if hasattr(session, 'absent_symptoms') else 'N/A'}")

print(f"\n=== Цикл питань ===")
for i in range(5):
    print(f"\n--- Ітерація {i+1} ---")
    
    # Перевіряємо should_continue
    should_cont = controller.should_continue()
    print(f"should_continue(): {should_cont}")
    
    if not should_cont:
        # Чому зупинились?
        if hasattr(controller, 'stop_decision'):
            sd = controller.stop_decision
            print(f"stop_decision: {sd}")
        if hasattr(controller, 'phase'):
            print(f"phase: {controller.phase}")
        if hasattr(controller, '_check_stopping'):
            # Спробуємо викликати перевірку
            print("Перевіряємо критерії зупинки вручну...")
        break
    
    q = controller.get_next_question()
    if not q:
        print("Питання = None")
        break
    
    print(f"Питання: {q.text}")
    print(f"Симптом: {q.symptom}, EIG: {q.eig:.4f}")
    
    # Відповідаємо
    answer = (i % 2 == 0)  # True, False, True...
    print(f"Відповідь: {'Так' if answer else 'Ні'}")
    
    r = controller.process_answer(answer)
    
    # Стан після відповіді
    print(f"Після відповіді:")
    print(f"  stop_decision: {r.stop_decision.reason.value if r.stop_decision else 'None'}")
    
    top3 = sorted(r.hypotheses.items(), key=lambda x: x[1], reverse=True)[:3]
    for d, p in top3:
        print(f"  {d}: {p:.1%}")

print(f"\n=== Фінал ===")
final = controller.get_result()
print(f"stop_reason: {final.stop_reason.value}")
print(f"iterations: {final.iterations}")
print(f"questions_asked: {final.questions_asked}")
print(f"present_symptoms: {final.present_symptoms}")
print(f"absent_symptoms: {final.absent_symptoms}")
