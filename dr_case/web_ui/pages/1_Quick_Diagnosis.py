"""
Dr.Case — Quick Diagnosis Page

Швидка діагностика: введи симптоми → отримай топ-10 діагнозів.
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Швидка діагностика — Dr.Case",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Швидка діагностика")
st.markdown("Введіть симптоми та отримайте список найімовірніших діагнозів.")

st.divider()

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

# Tabs для різних методів вводу
tab1, tab2 = st.tabs(["📝 Вибір симптомів", "💬 Текстовий опис"])

with tab1:
    st.subheader("Оберіть симптоми зі списку")
    
    # Multiselect для симптомів
    selected_symptoms = st.multiselect(
        "Симптоми:",
        options=symptoms_list,
        default=[],
        placeholder="Почніть вводити для пошуку...",
        help="Оберіть всі симптоми, які є у пацієнта"
    )
    
    # Швидкі кнопки для частих симптомів
    st.markdown("**Часті симптоми:**")
    
    frequent = ["Headache", "Fever", "Cough", "Fatigue", "Nausea", 
                "Abdominal Pain", "Chest Pain", "Shortness Of Breath"]
    
    cols = st.columns(4)
    for i, symptom in enumerate(frequent):
        if symptom in symptoms_list:
            if cols[i % 4].button(f"+ {symptom}", key=f"quick_{symptom}"):
                if symptom not in selected_symptoms:
                    selected_symptoms.append(symptom)
                    st.rerun()
    
    st.divider()
    
    # Кількість результатів
    top_k = st.slider("Кількість діагнозів:", min_value=5, max_value=20, value=10)
    
    # Кнопка діагностики
    if st.button("🔍 Діагностувати", type="primary", use_container_width=True, disabled=len(selected_symptoms) == 0):
        if selected_symptoms:
            with st.spinner("Аналіз симптомів..."):
                try:
                    start = time.time()
                    r = requests.post(
                        f"{API_URL}/api/diagnose",
                        json={"symptoms": selected_symptoms, "top_k": top_k}
                    )
                    elapsed = time.time() - start
                    
                    if r.status_code == 200:
                        data = r.json()
                        
                        st.success(f"✅ Аналіз завершено за {elapsed*1000:.0f}ms")
                        
                        # Відображення результатів
                        st.subheader("🎯 Результати діагностики")
                        
                        st.markdown(f"**Симптоми:** {', '.join(data.get('symptoms', []))}")
                        
                        st.divider()
                        
                        hypotheses = data.get("hypotheses", [])
                        
                        for h in hypotheses:
                            prob = h["probability"] * 100
                            rank = h["rank"]
                            disease = h["disease"]
                            
                            # Колір прогрес-бару
                            if prob >= 50:
                                color = "🟢"
                            elif prob >= 20:
                                color = "🟡"
                            else:
                                color = "🔵"
                            
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.markdown(f"### {rank}.")
                            with col2:
                                st.markdown(f"**{disease}**")
                                st.progress(prob / 100, text=f"{color} {prob:.1f}%")
                        
                    else:
                        st.error(f"Помилка: {r.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Помилка з'єднання: {e}")
        else:
            st.warning("Оберіть хоча б один симптом!")

with tab2:
    st.subheader("Опишіть симптоми текстом")
    
    st.info("💡 Підтримується українська та англійська мови")
    
    # Текстове поле
    text_input = st.text_area(
        "Опис скарг:",
        placeholder="Наприклад: Болить голова вже 3 дні, температура 38.5, кашель...",
        height=150
    )
    
    # Приклади
    st.markdown("**Приклади:**")
    examples = [
        "Болить голова і температура 38",
        "I have a cough and sore throat for 2 days",
        "Нудота, блювота, біль у животі",
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i].button(f"📝 {example[:25]}...", key=f"example_{i}"):
            text_input = example
            st.rerun()
    
    st.divider()
    
    top_k_text = st.slider("Кількість діагнозів:", min_value=5, max_value=20, value=10, key="top_k_text")
    
    if st.button("🔍 Аналізувати текст", type="primary", use_container_width=True, disabled=len(text_input.strip()) < 3):
        if text_input.strip():
            with st.spinner("Аналіз тексту..."):
                try:
                    # Спочатку витягуємо симптоми
                    r_extract = requests.post(
                        f"{API_URL}/api/symptoms/extract",
                        json={"text": text_input}
                    )
                    
                    if r_extract.status_code == 200:
                        extract_data = r_extract.json()
                        
                        symptoms = extract_data.get("symptoms", [])
                        vitals = extract_data.get("vitals", {})
                        duration = extract_data.get("duration", {})
                        
                        if symptoms:
                            # Показуємо що витягнуто
                            st.markdown("### 📋 Витягнуті дані")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Симптоми:**")
                                for s in symptoms:
                                    st.markdown(f"- {s}")
                            
                            with col2:
                                if vitals:
                                    st.markdown("**Вітальні показники:**")
                                    for k, v in vitals.items():
                                        st.markdown(f"- {k}: {v}")
                                else:
                                    st.markdown("**Вітальні показники:**")
                                    st.caption("Не виявлено")
                            
                            with col3:
                                if duration:
                                    st.markdown("**Тривалість:**")
                                    for k, v in duration.items():
                                        st.markdown(f"- {v} {k}")
                                else:
                                    st.markdown("**Тривалість:**")
                                    st.caption("Не вказано")
                            
                            st.divider()
                            
                            # Тепер діагностика
                            r = requests.post(
                                f"{API_URL}/api/diagnose",
                                json={"symptoms": symptoms, "top_k": top_k_text}
                            )
                            
                            if r.status_code == 200:
                                data = r.json()
                                
                                st.subheader("🎯 Результати діагностики")
                                
                                hypotheses = data.get("hypotheses", [])
                                
                                for h in hypotheses:
                                    prob = h["probability"] * 100
                                    rank = h["rank"]
                                    disease = h["disease"]
                                    
                                    col1, col2 = st.columns([1, 4])
                                    with col1:
                                        st.markdown(f"### {rank}.")
                                    with col2:
                                        st.markdown(f"**{disease}**")
                                        st.progress(prob / 100, text=f"{prob:.1f}%")
                        else:
                            st.warning("Симптоми не знайдено в тексті. Спробуйте описати детальніше.")
                    else:
                        st.error("Помилка витягування симптомів")
                        
                except Exception as e:
                    st.error(f"Помилка: {e}")

# Footer
st.divider()
st.caption("⚠️ Ця система не замінює консультацію з лікарем. Завжди звертайтесь до спеціалістів.")
