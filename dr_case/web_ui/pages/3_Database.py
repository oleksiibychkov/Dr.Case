"""
Dr.Case — Database Explorer Page

Перегляд бази даних симптомів та хвороб.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="База даних — Dr.Case",
    page_icon="📊",
    layout="wide",
)

st.title("📊 База даних")
st.markdown("Перегляд симптомів та хвороб у системі.")

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


tab1, tab2 = st.tabs(["🩺 Симптоми", "🦠 Хвороби"])

with tab1:
    st.subheader("🩺 Пошук симптомів")
    
    search_query = st.text_input(
        "Пошук:",
        placeholder="Введіть назву симптому...",
        key="symptom_search"
    )
    
    if search_query:
        try:
            r = requests.get(f"{API_URL}/api/symptoms/search?q={search_query}&limit=50")
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                
                st.markdown(f"**Знайдено:** {data.get('total', 0)} результатів")
                
                for s in results:
                    st.markdown(f"- {s['name']}")
            else:
                st.warning("Нічого не знайдено")
        except Exception as e:
            st.error(f"Помилка: {e}")
    else:
        # Показуємо всі симптоми
        st.markdown("**Всі симптоми:**")
        
        try:
            r = requests.get(f"{API_URL}/api/symptoms?limit=500")
            if r.status_code == 200:
                symptoms = [s["name"] for s in r.json()]
                
                # Групуємо по першій літері
                letters = sorted(set(s[0].upper() for s in symptoms))
                
                selected_letter = st.selectbox("Фільтр по літері:", ["Всі"] + letters)
                
                if selected_letter != "Всі":
                    symptoms = [s for s in symptoms if s[0].upper() == selected_letter]
                
                st.markdown(f"**Показано:** {len(symptoms)} симптомів")
                
                # Відображаємо в колонках
                cols = st.columns(3)
                for i, symptom in enumerate(symptoms):
                    cols[i % 3].markdown(f"• {symptom}")
                    
        except Exception as e:
            st.error(f"Помилка: {e}")

with tab2:
    st.subheader("🦠 Пошук хвороб")
    
    disease_search = st.text_input(
        "Пошук:",
        placeholder="Введіть назву хвороби...",
        key="disease_search"
    )
    
    try:
        r = requests.get(f"{API_URL}/api/diseases?limit=1000")
        if r.status_code == 200:
            data = r.json()
            all_diseases = data.get("diseases", [])
            
            if disease_search:
                filtered = [d for d in all_diseases if disease_search.lower() in d.lower()]
            else:
                filtered = all_diseases
            
            st.markdown(f"**Показано:** {len(filtered)} з {len(all_diseases)} хвороб")
            
            # Групуємо по першій літері
            letters = sorted(set(d[0].upper() for d in filtered if d))
            
            selected_letter = st.selectbox("Фільтр по літері:", ["Всі"] + letters, key="disease_letter")
            
            if selected_letter != "Всі":
                filtered = [d for d in filtered if d[0].upper() == selected_letter]
            
            # Пагінація
            page_size = 50
            total_pages = (len(filtered) + page_size - 1) // page_size
            
            if total_pages > 1:
                page = st.number_input("Сторінка:", min_value=1, max_value=total_pages, value=1)
            else:
                page = 1
            
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            diseases_page = filtered[start_idx:end_idx]
            
            # Відображаємо
            for disease in diseases_page:
                with st.expander(f"🦠 {disease}"):
                    # Отримуємо деталі
                    try:
                        r_detail = requests.get(f"{API_URL}/api/diseases/{disease}")
                        if r_detail.status_code == 200:
                            detail = r_detail.json()
                            symptoms = detail.get("symptoms", [])
                            
                            st.markdown(f"**Кількість симптомів:** {len(symptoms)}")
                            st.markdown("**Симптоми:**")
                            
                            # В колонках
                            cols = st.columns(3)
                            for i, s in enumerate(symptoms):
                                cols[i % 3].markdown(f"• {s}")
                        else:
                            st.info("Деталі недоступні")
                    except:
                        st.info("Не вдалося завантажити деталі")
                        
    except Exception as e:
        st.error(f"Помилка: {e}")

# Footer
st.divider()
st.caption("Dr.Case — База даних медичних діагнозів")
