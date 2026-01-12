"""
Dr.Case — Standalone Streamlit Application for Streamlit Cloud

Ця версія працює БЕЗ окремого API сервера.
Завантажує HouseFlowEngine напряму.

Deploy: https://share.streamlit.io
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Додаємо шлях до проекту
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Dr.Case — Медична діагностика",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Завантаження Engine (кешується)
# ============================================================================

@st.cache_resource(show_spinner="Завантаження моделей...")
def load_engine():
    """Завантажити HouseFlowEngine один раз"""
    try:
        from dr_case.diagnosis_engine.house_flow_engine import HouseFlowEngine
        
        # Шляхи до файлів
        som_path = PROJECT_ROOT / "models" / "som_model.pkl"
        nn_path = PROJECT_ROOT / "models" / "nn_two_branch.pt"
        db_path = PROJECT_ROOT / "data" / "unified_disease_symptom_data_full.json"
        
        # Альтернативні шляхи
        if not som_path.exists():
            som_path = PROJECT_ROOT / "models" / "som_merged.pkl"
        if not som_path.exists():
            som_path = PROJECT_ROOT / "models" / "som_optimized.pkl"
        
        engine = HouseFlowEngine.load(
            som_model_path=str(som_path) if som_path.exists() else None,
            nn_model_path=str(nn_path) if nn_path.exists() else None,
            database_path=str(db_path) if db_path.exists() else None,
        )
        return engine, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_database():
    """Завантажити базу даних для пошуку симптомів"""
    import json
    db_path = PROJECT_ROOT / "data" / "unified_disease_symptom_data_full.json"
    
    if db_path.exists():
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Збираємо всі симптоми
        symptoms = set()
        for disease, info in data.items():
            if isinstance(info, dict) and 'symptoms' in info:
                symptoms.update(info['symptoms'])
            elif isinstance(info, list):
                symptoms.update(info)
        
        return list(sorted(symptoms)), data
    return [], {}


# ============================================================================
# Головна сторінка
# ============================================================================

def main():
    # Sidebar
    st.sidebar.title("🏥 Dr.Case")
    st.sidebar.markdown("Система медичної діагностики")
    
    page = st.sidebar.radio(
        "Навігація",
        ["🏠 Головна", "🔍 Швидка діагностика", "💬 Інтерактивна", "📊 База даних", "ℹ️ Про систему"]
    )
    
    if page == "🏠 Головна":
        show_home()
    elif page == "🔍 Швидка діагностика":
        show_quick_diagnosis()
    elif page == "💬 Інтерактивна":
        show_interactive()
    elif page == "📊 База даних":
        show_database()
    elif page == "ℹ️ Про систему":
        show_about()


def show_home():
    """Головна сторінка"""
    st.title("🏥 Dr.Case")
    st.markdown("### Інтелектуальна система медичної діагностики")
    
    st.markdown("""
    **Dr.Case** — це система диференціальної медичної діагностики, що використовує:
    
    - 🧠 **Self-Organizing Map (SOM)** — для визначення клінічного сценарію
    - 🤖 **Neural Network** — для ранжування діагнозів
    - 💬 **NLP** — для розуміння природної мови
    
    ---
    
    ### 🚀 Як почати?
    
    1. **Швидка діагностика** — введіть симптоми та отримайте топ-10 діагнозів
    2. **Інтерактивна сесія** — покрокова діагностика з уточнюючими питаннями
    
    Оберіть потрібну сторінку в меню зліва 👈
    """)
    
    st.divider()
    
    # Статистика
    st.subheader("📊 Статистика бази")
    
    symptoms_list, db = load_database()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🦠 Хвороб", len(db))
    col2.metric("🩺 Симптомів", len(symptoms_list))
    col3.metric("📈 Accuracy", "~91%")
    col4.metric("⚡ Модель", "SOM + NN")
    
    st.divider()
    
    st.warning("""
    ⚠️ **Важливо**
    
    Ця система призначена лише для **освітніх та дослідницьких цілей**.  
    Вона **НЕ ЗАМІНЮЄ** консультацію з кваліфікованим лікарем.  
    Завжди звертайтесь до медичних спеціалістів для діагностики та лікування.
    """)
    
    st.caption("*Розроблено: Oleksii Bychkov, Taras Shevchenko National University of Kyiv*")


def show_quick_diagnosis():
    """Швидка діагностика"""
    st.title("🔍 Швидка діагностика")
    st.markdown("Введіть симптоми та отримайте список найімовірніших діагнозів.")
    
    # Завантажуємо engine
    engine, error = load_engine()
    
    if error:
        st.error(f"❌ Помилка завантаження: {error}")
        st.info("Перевірте наявність файлів моделей у папці `models/`")
        return
    
    if engine is None:
        st.error("❌ Engine не завантажено")
        return
    
    st.divider()
    
    # Вибір способу вводу
    input_method = st.radio(
        "Спосіб вводу:",
        ["📝 Вибір зі списку", "💬 Текстовий опис"],
        horizontal=True
    )
    
    symptoms_list, _ = load_database()
    
    if input_method == "📝 Вибір зі списку":
        # Вибір симптомів зі списку
        selected_symptoms = st.multiselect(
            "Оберіть симптоми:",
            options=symptoms_list,
            max_selections=20,
            placeholder="Почніть вводити назву симптому..."
        )
        
        # Швидкий вибір частих симптомів
        st.markdown("**Часті симптоми:**")
        cols = st.columns(6)
        frequent = ["Fever", "Headache", "Cough", "Fatigue", "Nausea", "Dizziness"]
        
        for i, symptom in enumerate(frequent):
            if symptom in symptoms_list:
                if cols[i].button(f"+ {symptom}", key=f"freq_{symptom}"):
                    if symptom not in selected_symptoms:
                        selected_symptoms.append(symptom)
                        st.rerun()
        
        symptoms_to_diagnose = selected_symptoms
        
    else:
        # Текстовий опис
        text_input = st.text_area(
            "Опишіть симптоми:",
            placeholder="Наприклад: Болить голова, висока температура, кашель...",
            height=100
        )
        
        symptoms_to_diagnose = []
        
        if text_input:
            try:
                from dr_case.nlp import extract_symptoms
                result = extract_symptoms(text_input)
                symptoms_to_diagnose = result.present
                
                if symptoms_to_diagnose:
                    st.success(f"✅ Знайдено симптоми: {', '.join(symptoms_to_diagnose)}")
                if result.absent:
                    st.info(f"ℹ️ Відсутні симптоми: {', '.join(result.absent)}")
            except ImportError:
                st.warning("⚠️ NLP модуль недоступний. Використовуйте вибір зі списку.")
            except Exception as e:
                st.error(f"Помилка NLP: {e}")
    
    st.divider()
    
    # Кнопка діагностики
    if st.button("🔬 Діагностувати", type="primary", disabled=not symptoms_to_diagnose):
        with st.spinner("Аналіз симптомів..."):
            try:
                # Створюємо сесію і отримуємо діагноз
                session = engine.start_session(symptoms_to_diagnose)
                
                # Отримуємо поточні гіпотези
                hypotheses = session.current_hypotheses[:10]
                
                st.success("✅ Аналіз завершено!")
                
                st.subheader("📋 Топ-10 діагнозів")
                
                for i, hyp in enumerate(hypotheses, 1):
                    disease_name = hyp.disease_name if hasattr(hyp, 'disease_name') else hyp.get('disease', 'Unknown')
                    prob = hyp.probability if hasattr(hyp, 'probability') else hyp.get('probability', 0)
                    prob_pct = prob * 100 if prob <= 1 else prob
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**{i}.**")
                    with col2:
                        st.markdown(f"**{disease_name}**")
                        st.progress(min(prob_pct / 100, 1.0), text=f"{prob_pct:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Помилка: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer
    st.divider()
    st.caption("⚠️ Ця система не замінює консультацію з лікарем.")


def show_interactive():
    """Інтерактивна діагностика"""
    st.title("💬 Інтерактивна діагностика")
    st.markdown("Покрокова діагностика з уточнюючими питаннями.")
    
    engine, error = load_engine()
    
    if error:
        st.error(f"❌ Помилка: {error}")
        return
    
    if engine is None:
        st.error("❌ Engine не завантажено")
        return
    
    # Session state для інтерактивної сесії
    if "interactive_session" not in st.session_state:
        st.session_state.interactive_session = None
    if "interactive_history" not in st.session_state:
        st.session_state.interactive_history = []
    
    st.divider()
    
    # Якщо немає активної сесії - створюємо
    if st.session_state.interactive_session is None:
        symptoms_list, _ = load_database()
        
        st.subheader("🚀 Почати нову сесію")
        
        initial_symptoms = st.multiselect(
            "Початкові симптоми:",
            options=symptoms_list,
            max_selections=10,
            placeholder="Оберіть хоча б один симптом..."
        )
        
        if st.button("▶️ Почати діагностику", disabled=not initial_symptoms):
            try:
                session = engine.start_session(initial_symptoms)
                st.session_state.interactive_session = session
                st.session_state.interactive_history = [
                    {"type": "start", "symptoms": initial_symptoms}
                ]
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")
    
    else:
        # Активна сесія
        session = st.session_state.interactive_session
        
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.subheader("📊 Поточні гіпотези")
            
            hypotheses = session.current_hypotheses[:5]
            for i, hyp in enumerate(hypotheses, 1):
                disease_name = hyp.disease_name if hasattr(hyp, 'disease_name') else str(hyp)
                prob = hyp.probability if hasattr(hyp, 'probability') else 0
                prob_pct = prob * 100 if prob <= 1 else prob
                
                st.markdown(f"**{i}. {disease_name}** — {prob_pct:.1f}%")
                st.progress(min(prob_pct / 100, 1.0))
            
            st.divider()
            
            # Перевіряємо чи є питання
            if hasattr(session, 'current_question') and session.current_question:
                question = session.current_question
                
                st.subheader("❓ Питання")
                st.markdown(f"**{question}**")
                
                col1, col2, col3 = st.columns(3)
                
                if col1.button("✅ Так", use_container_width=True):
                    try:
                        engine.answer_question(session, True)
                        st.session_state.interactive_history.append({
                            "type": "answer",
                            "question": question,
                            "answer": True
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Помилка: {e}")
                
                if col2.button("❌ Ні", use_container_width=True):
                    try:
                        engine.answer_question(session, False)
                        st.session_state.interactive_history.append({
                            "type": "answer",
                            "question": question,
                            "answer": False
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Помилка: {e}")
                
                if col3.button("🤷 Не знаю", use_container_width=True):
                    try:
                        engine.answer_question(session, None)
                        st.session_state.interactive_history.append({
                            "type": "answer",
                            "question": question,
                            "answer": None
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Помилка: {e}")
            else:
                st.success("✅ Діагностика завершена!")
                
                if hypotheses:
                    top = hypotheses[0]
                    disease_name = top.disease_name if hasattr(top, 'disease_name') else str(top)
                    st.markdown(f"### 🎯 Найімовірніший діагноз: **{disease_name}**")
        
        with col_side:
            st.subheader("📜 Історія")
            
            for item in st.session_state.interactive_history:
                if item["type"] == "start":
                    st.markdown(f"🚀 Початок: {len(item['symptoms'])} симптомів")
                elif item["type"] == "answer":
                    ans = "Так" if item["answer"] is True else "Ні" if item["answer"] is False else "Не знаю"
                    st.markdown(f"❓ {item['question'][:30]}... → **{ans}**")
            
            st.divider()
            
            if st.button("🔄 Нова сесія"):
                st.session_state.interactive_session = None
                st.session_state.interactive_history = []
                st.rerun()
    
    st.divider()
    st.caption("⚠️ Ця система не замінює консультацію з лікарем.")


def show_database():
    """База даних"""
    st.title("📊 База даних")
    st.markdown("Перегляд симптомів та хвороб у системі.")
    
    symptoms_list, db = load_database()
    
    st.divider()
    
    tab1, tab2 = st.tabs(["🦠 Хвороби", "🩺 Симптоми"])
    
    with tab1:
        st.subheader(f"Всього хвороб: {len(db)}")
        
        search_disease = st.text_input("🔍 Пошук хвороби:", placeholder="Введіть назву...")
        
        diseases = list(db.keys())
        if search_disease:
            diseases = [d for d in diseases if search_disease.lower() in d.lower()]
        
        # Пагінація
        per_page = 20
        total_pages = (len(diseases) - 1) // per_page + 1
        page = st.number_input("Сторінка", 1, total_pages, 1) if total_pages > 1 else 1
        
        start = (page - 1) * per_page
        end = start + per_page
        
        for disease in diseases[start:end]:
            with st.expander(disease):
                info = db[disease]
                if isinstance(info, dict) and 'symptoms' in info:
                    symptoms = info['symptoms']
                elif isinstance(info, list):
                    symptoms = info
                else:
                    symptoms = []
                
                st.markdown(f"**Симптоми ({len(symptoms)}):**")
                st.write(", ".join(symptoms[:20]))
                if len(symptoms) > 20:
                    st.write(f"... та ще {len(symptoms) - 20}")
    
    with tab2:
        st.subheader(f"Всього симптомів: {len(symptoms_list)}")
        
        search_symptom = st.text_input("🔍 Пошук симптому:", placeholder="Введіть назву...")
        
        filtered = symptoms_list
        if search_symptom:
            filtered = [s for s in symptoms_list if search_symptom.lower() in s.lower()]
        
        # Показуємо в колонках
        cols = st.columns(3)
        for i, symptom in enumerate(filtered[:60]):
            cols[i % 3].write(f"• {symptom}")
        
        if len(filtered) > 60:
            st.info(f"Показано 60 з {len(filtered)} симптомів")


def show_about():
    """Про систему"""
    st.title("ℹ️ Про систему Dr.Case")
    
    st.markdown("""
    ## 🎯 Призначення
    
    **Dr.Case** — інтелектуальна система диференціальної медичної діагностики,
    розроблена як дипломний проект.
    
    ---
    
    ## 🏗️ Архітектура
    
    | Компонент | Призначення |
    |-----------|-------------|
    | **SOM (Self-Organizing Map)** | Кластеризація хвороб за симптомами |
    | **Neural Network** | Ранжування діагнозів |
    | **NLP Module** | Витягування симптомів з тексту |
    | **Question Engine** | Вибір уточнюючих питань |
    
    ---
    
    ## 📊 База даних
    
    - **844 захворювання**
    - **460+ симптомів**
    - Джерело: Unified Disease-Symptom Database
    
    ---
    
    ## 🔬 Методологія
    
    Система використовує **циклічний процес діагностики** (як у серіалі "Доктор Хаус"):
    
    1. Збір симптомів
    2. Генерація гіпотез
    3. Уточнюючі питання
    4. Звуження диференціалу
    5. Фінальний діагноз
    
    ---
    
    ## ⚠️ Застереження
    
    > **Ця система призначена ВИКЛЮЧНО для освітніх та дослідницьких цілей.**
    > 
    > Вона **НЕ ЗАМІНЮЄ** консультацію з кваліфікованим лікарем.
    > 
    > Завжди звертайтесь до медичних спеціалістів.
    
    ---
    
    ## 👨‍💻 Розробник
    
    **Oleksii Bychkov**  
    Taras Shevchenko National University of Kyiv
    
    ---
    
    ## 📚 Технології
    
    - Python 3.10+
    - PyTorch (Neural Networks)
    - MiniSOM (Self-Organizing Maps)
    - Streamlit (Web UI)
    - FastAPI (REST API)
    
    ---
    
    *Версія: 1.0.0*
    """)


if __name__ == "__main__":
    main()
