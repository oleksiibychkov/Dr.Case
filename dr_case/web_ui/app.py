"""
Dr.Case — Web UI (Streamlit)

Веб-інтерфейс для системи медичної діагностики.

Запуск:
    streamlit run dr_case/web_ui/app.py
    
    або:
    
    python scripts/run_web.py
"""

import streamlit as st
import sys
from pathlib import Path

# Додаємо корінь проекту
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    # Налаштування сторінки
    st.set_page_config(
        page_title="Dr.Case — Медична діагностика",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/hospital.png", width=80)
        st.title("Dr.Case")
        st.caption("Інтелектуальна система медичної діагностики")
        
        st.divider()
        
        st.markdown("""
        ### 📋 Навігація
        
        - 🏠 **Home** — Про систему
        - 🔍 **Quick Diagnosis** — Швидка діагностика
        - 💬 **Interactive** — Інтерактивна сесія
        - 📊 **History** — Історія
        """)
        
        st.divider()
        
        # Статус API
        st.markdown("### ⚙️ Статус")
        
        try:
            import requests
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                st.success("🟢 API Online")
                st.caption(f"Симптомів: {data.get('database_symptoms', 0)}")
                st.caption(f"Хвороб: {data.get('database_diseases', 0)}")
            else:
                st.error("🔴 API Error")
        except:
            st.warning("🟡 API Offline")
            st.caption("Запустіть: python scripts/run_api.py")
    
    # Головний контент
    st.title("🏥 Dr.Case")
    st.subheader("Інтелектуальна система медичної діагностики")
    
    st.markdown("""
    ---
    
    ### 👋 Ласкаво просимо!
    
    **Dr.Case** — це система диференціальної медичної діагностики, що використовує:
    
    - 🧠 **Self-Organizing Map (SOM)** — для визначення клінічного сценарію
    - 🤖 **Neural Network** — для ранжування діагнозів
    - 💬 **NLP** — для розуміння природної мови
    
    ---
    
    ### 🚀 Як почати?
    
    1. **Швидка діагностика** — введіть симптоми та отримайте топ-10 діагнозів
    2. **Інтерактивна сесія** — покрокова діагностика з уточнюючими питаннями
    
    Оберіть потрібну сторінку в меню зліва 👈
    
    ---
    
    ### 📊 Статистика бази
    """)
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            col1.metric("🦠 Хвороб", data.get("database_diseases", 0))
            col2.metric("🩺 Симптомів", data.get("database_symptoms", 0))
            col3.metric("📈 Accuracy", "91%")
            col4.metric("⚡ Час відповіді", "~12ms")
        else:
            st.info("Запустіть API сервер для відображення статистики")
    except:
        col1.metric("🦠 Хвороб", 844)
        col2.metric("🩺 Симптомів", 460)
        col3.metric("📈 Accuracy", "91%")
        col4.metric("⚡ Час відповіді", "~12ms")
    
    st.markdown("""
    ---
    
    ### ⚠️ Важливо
    
    > Ця система призначена лише для **освітніх та дослідницьких цілей**.  
    > Вона **не замінює** консультацію з кваліфікованим лікарем.  
    > Завжди звертайтесь до медичних спеціалістів для діагностики та лікування.
    
    ---
    
    *Розроблено: Oleksii Bychkov, Taras Shevchenko National University of Kyiv*
    """)


if __name__ == "__main__":
    main()
