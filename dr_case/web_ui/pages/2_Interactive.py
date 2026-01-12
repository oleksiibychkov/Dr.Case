"""
Dr.Case — Interactive Diagnosis Page

Інтерактивна діагностика з покроковими питаннями.
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Інтерактивна діагностика — Dr.Case",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Інтерактивна діагностика")
st.markdown("Покрокова діагностика з уточнюючими питаннями.")

st.divider()

# Ініціалізація session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_data" not in st.session_state:
    st.session_state.session_data = None
if "history" not in st.session_state:
    st.session_state.history = []

# Перевірка API
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

if not check_api():
    st.error("❌ API сервер недоступний!")
    st.info("Запустіть: `python scripts/run_api.py`")
    st.stop()

# Отримуємо список симптомів
@st.cache_data(ttl=300)
def get_symptoms():
    try:
        r = requests.get(f"{API_URL}/api/symptoms?limit=500")
        return [s["name"] for s in r.json()]
    except:
        return []

symptoms_list = get_symptoms()


def create_session(symptoms: list):
    """Створити нову сесію"""
    try:
        r = requests.post(
            f"{API_URL}/api/sessions",
            json={"symptoms": symptoms, "language": "uk"}
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def get_session(session_id: str):
    """Отримати стан сесії"""
    try:
        r = requests.get(f"{API_URL}/api/sessions/{session_id}")
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def answer_question(session_id: str, answer: bool):
    """Відповісти на питання"""
    try:
        r = requests.post(
            f"{API_URL}/api/sessions/{session_id}/answer",
            json={"answer": answer}
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def delete_session(session_id: str):
    """Видалити сесію"""
    try:
        requests.delete(f"{API_URL}/api/sessions/{session_id}")
    except:
        pass


# Головний інтерфейс
if st.session_state.session_id is None:
    # ========== СТВОРЕННЯ СЕСІЇ ==========
    st.subheader("🆕 Нова діагностика")
    
    st.markdown("Оберіть початкові симптоми пацієнта:")
    
    # Multiselect
    initial_symptoms = st.multiselect(
        "Симптоми:",
        options=symptoms_list,
        default=[],
        placeholder="Оберіть симптоми...",
    )
    
    # Або текстовий ввід
    st.markdown("**Або опишіть текстом:**")
    text_input = st.text_input(
        "Опис скарг:",
        placeholder="Болить голова, температура 38..."
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Почати діагностику", type="primary", use_container_width=True):
            symptoms_to_use = initial_symptoms
            
            # Якщо є текст — витягуємо симптоми
            if text_input.strip() and not initial_symptoms:
                try:
                    r = requests.post(
                        f"{API_URL}/api/symptoms/extract",
                        json={"text": text_input}
                    )
                    if r.status_code == 200:
                        symptoms_to_use = r.json().get("symptoms", [])
                except:
                    pass
            
            if symptoms_to_use:
                session_data = create_session(symptoms_to_use)
                
                if session_data:
                    st.session_state.session_id = session_data["session_id"]
                    st.session_state.session_data = session_data
                    st.session_state.history = [{
                        "type": "start",
                        "symptoms": symptoms_to_use
                    }]
                    st.rerun()
                else:
                    st.error("Не вдалося створити сесію")
            else:
                st.warning("Оберіть хоча б один симптом!")
    
    with col2:
        st.button("🔄 Скинути", use_container_width=True, disabled=True)

else:
    # ========== АКТИВНА СЕСІЯ ==========
    session_data = st.session_state.session_data
    
    # Sidebar з інформацією
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📋 Поточна сесія")
        st.markdown(f"**ID:** `{st.session_state.session_id}`")
        st.markdown(f"**Статус:** {session_data.get('status', 'unknown')}")
        st.markdown(f"**Ітерація:** {session_data.get('iteration', 0)}")
        
        st.markdown("---")
        st.markdown("**Підтверджені симптоми:**")
        for s in session_data.get("confirmed_symptoms", []):
            st.markdown(f"✅ {s}")
        
        if session_data.get("denied_symptoms"):
            st.markdown("**Заперечені симптоми:**")
            for s in session_data.get("denied_symptoms", []):
                st.markdown(f"❌ {s}")
        
        st.markdown("---")
        if st.button("🗑️ Завершити сесію", use_container_width=True):
            delete_session(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.session_data = None
            st.session_state.history = []
            st.rerun()
    
    # Основний контент
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("🎯 Гіпотези")
        
        hypotheses = session_data.get("hypotheses", [])
        
        if hypotheses:
            for h in hypotheses[:7]:
                prob = h.get("probability", 0) * 100
                disease = h.get("disease", "Unknown")
                rank = h.get("rank", 0)
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"**#{rank}**")
                with col2:
                    st.progress(prob / 100, text=f"{disease} — {prob:.1f}%")
        else:
            st.info("Гіпотези ще не сформовані")
        
        st.divider()
        
        # Питання або результат
        if session_data.get("status") == "waiting_answer" and session_data.get("current_question"):
            question = session_data["current_question"]
            symptom = question.get("symptom", "")
            text_uk = question.get("text_uk", f"Чи є у вас {symptom}?")
            
            st.subheader("❓ Уточнююче питання")
            
            st.markdown(f"### {text_uk}")
            
            st.markdown(f"*Симптом: {symptom}*")
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Так", type="primary", use_container_width=True):
                    result = answer_question(st.session_state.session_id, True)
                    if result:
                        st.session_state.session_data = result.get("session_state", session_data)
                        st.session_state.history.append({
                            "type": "answer",
                            "question": symptom,
                            "answer": True
                        })
                        st.rerun()
            
            with col2:
                if st.button("❌ Ні", use_container_width=True):
                    result = answer_question(st.session_state.session_id, False)
                    if result:
                        st.session_state.session_data = result.get("session_state", session_data)
                        st.session_state.history.append({
                            "type": "answer",
                            "question": symptom,
                            "answer": False
                        })
                        st.rerun()
            
            with col3:
                if st.button("🤷 Не знаю", use_container_width=True):
                    result = answer_question(st.session_state.session_id, None)
                    if result:
                        st.session_state.session_data = result.get("session_state", session_data)
                        st.session_state.history.append({
                            "type": "answer",
                            "question": symptom,
                            "answer": None
                        })
                        st.rerun()
        
        elif session_data.get("status") == "completed":
            st.subheader("✅ Діагностику завершено")
            
            final = session_data.get("final_diagnosis")
            reason = session_data.get("stop_reason")
            
            if final:
                st.success(f"**Найімовірніший діагноз:** {final}")
            
            if reason:
                st.info(f"Причина завершення: {reason}")
            
            st.divider()
            
            st.markdown("### 📋 Рекомендації")
            st.markdown("""
            1. Зверніться до лікаря для підтвердження діагнозу
            2. Не займайтесь самолікуванням
            3. При погіршенні стану викликайте швидку допомогу
            """)
            
            if st.button("🆕 Нова діагностика", type="primary"):
                delete_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.session_data = None
                st.session_state.history = []
                st.rerun()
        
        else:
            st.info(f"Статус: {session_data.get('status', 'active')}")
            
            # Кнопка оновлення
            if st.button("🔄 Оновити стан"):
                new_data = get_session(st.session_state.session_id)
                if new_data:
                    st.session_state.session_data = new_data
                    st.rerun()
    
    with col_side:
        st.subheader("📜 Історія")
        
        for item in reversed(st.session_state.history):
            if item["type"] == "start":
                st.markdown(f"🚀 Початок: {len(item['symptoms'])} симптомів")
            elif item["type"] == "answer":
                answer_text = "Так" if item["answer"] is True else "Ні" if item["answer"] is False else "Не знаю"
                st.markdown(f"❓ {item['question']}: **{answer_text}**")

# Footer
st.divider()
st.caption("⚠️ Ця система не замінює консультацію з лікарем.")
