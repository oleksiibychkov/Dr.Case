"""
Dr.Case — Повністю автономний Streamlit Application

Ця версія НЕ залежить від внутрішніх модулів dr_case.
Завантажує SOM та NN моделі напряму.

Deploy: https://share.streamlit.io
"""

import streamlit as st
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# Конфігурація
# ============================================================================

st.set_page_config(
    page_title="Dr.Case — Медична діагностика",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class DiagnosisResult:
    disease_name: str
    probability: float
    matching_symptoms: List[str]


# ============================================================================
# Завантаження даних (кешується)
# ============================================================================

@st.cache_data
def load_database() -> Tuple[Dict, List[str], Dict[str, int]]:
    """Завантажити базу даних захворювань"""
    db_paths = [
        PROJECT_ROOT / "data" / "unified_disease_symptom_merged.json",
        PROJECT_ROOT / "data" / "unified_disease_symptom_data_full.json",
    ]
    
    db = {}
    for path in db_paths:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            break
    
    if not db:
        return {}, [], {}
    
    # Збираємо всі симптоми
    all_symptoms = set()
    for disease, info in db.items():
        if isinstance(info, dict) and 'symptoms' in info:
            all_symptoms.update(info['symptoms'])
        elif isinstance(info, list):
            all_symptoms.update(info)
    
    symptoms_list = sorted(list(all_symptoms))
    symptom_to_idx = {s: i for i, s in enumerate(symptoms_list)}
    
    return db, symptoms_list, symptom_to_idx


@st.cache_resource
def load_som_model():
    """Завантажити SOM модель"""
    som_paths = [
        PROJECT_ROOT / "models" / "som_model.pkl",
        PROJECT_ROOT / "models" / "som_merged.pkl",
        PROJECT_ROOT / "models" / "som_optimized.pkl",
    ]
    
    for path in som_paths:
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                return data, None
            except Exception as e:
                return None, str(e)
    
    return None, "SOM model not found"


@st.cache_resource
def load_nn_model():
    """Завантажити Neural Network модель"""
    try:
        import torch
        import torch.nn as nn
        
        nn_paths = [
            PROJECT_ROOT / "models" / "nn_two_branch.pt",
            PROJECT_ROOT / "models" / "nn_model.pt",
            PROJECT_ROOT / "models" / "nn_model_pseudo.pt",
        ]
        
        for path in nn_paths:
            if path.exists():
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                return checkpoint, None
        
        return None, "NN model not found"
    except ImportError:
        return None, "PyTorch not available"
    except Exception as e:
        return None, str(e)


class TwoBranchNN:
    """Wrapper для Two-Branch Neural Network"""
    
    def __init__(self, checkpoint, n_symptoms: int, n_diseases: int):
        try:
            import torch
            import torch.nn as nn
            
            self.device = torch.device('cpu')
            self.n_symptoms = n_symptoms
            self.n_diseases = n_diseases
            
            # Отримуємо параметри з checkpoint
            if isinstance(checkpoint, dict):
                self.disease_to_idx = checkpoint.get('disease_to_idx', {})
                self.idx_to_disease = checkpoint.get('idx_to_disease', {})
                self.symptom_to_idx = checkpoint.get('symptom_to_idx', {})
                
                # Визначаємо архітектуру
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
                
                if state_dict:
                    # Визначаємо розміри з state_dict
                    self.model = self._build_model_from_state(state_dict)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.ready = True
                else:
                    self.ready = False
            else:
                self.ready = False
                
        except Exception as e:
            self.ready = False
            self.error = str(e)
    
    def _build_model_from_state(self, state_dict):
        """Побудувати модель на основі state_dict"""
        import torch
        import torch.nn as nn
        
        # Спробувати визначити архітектуру з ключів
        keys = list(state_dict.keys())
        
        # Простий MLP
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, h_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.3))
                    prev_dim = h_dim
                layers.append(nn.Linear(prev_dim, output_dim))
                layers.append(nn.Sigmoid())
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        # Визначаємо розміри з першого шару
        first_weight_key = [k for k in keys if 'weight' in k][0]
        first_weight = state_dict[first_weight_key]
        input_dim = first_weight.shape[1]
        
        # Визначаємо output з останнього шару
        last_weight_key = [k for k in keys if 'weight' in k][-1]
        last_weight = state_dict[last_weight_key]
        output_dim = last_weight.shape[0]
        
        # Визначаємо hidden dims
        hidden_dims = []
        for k in keys:
            if 'weight' in k and k != first_weight_key and k != last_weight_key:
                hidden_dims.append(state_dict[k].shape[0])
        
        if not hidden_dims:
            hidden_dims = [256, 128]
        
        return SimpleMLP(input_dim, hidden_dims, output_dim)
    
    def predict(self, symptom_vector: np.ndarray) -> Dict[str, float]:
        """Передбачити ймовірності діагнозів"""
        if not self.ready:
            return {}
        
        try:
            import torch
            
            with torch.no_grad():
                x = torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0)
                probs = self.model(x).squeeze().numpy()
            
            # Конвертуємо в словник
            results = {}
            for idx, prob in enumerate(probs):
                disease = self.idx_to_disease.get(idx, self.idx_to_disease.get(str(idx), f"Disease_{idx}"))
                results[disease] = float(prob)
            
            return results
            
        except Exception as e:
            return {}


# ============================================================================
# Діагностичний движок (спрощений)
# ============================================================================

class SimpleDiagnosisEngine:
    """Спрощений діагностичний движок без складних залежностей"""
    
    def __init__(self, db: Dict, symptoms_list: List[str], symptom_to_idx: Dict[str, int]):
        self.db = db
        self.symptoms_list = symptoms_list
        self.symptom_to_idx = symptom_to_idx
        self.n_symptoms = len(symptoms_list)
        
        # Побудувати індекс disease -> symptoms set
        self.disease_symptoms = {}
        for disease, info in db.items():
            if isinstance(info, dict) and 'symptoms' in info:
                self.disease_symptoms[disease] = set(info['symptoms'])
            elif isinstance(info, list):
                self.disease_symptoms[disease] = set(info)
            else:
                self.disease_symptoms[disease] = set()
    
    def encode_symptoms(self, symptoms: List[str]) -> np.ndarray:
        """Перетворити список симптомів у бінарний вектор"""
        vector = np.zeros(self.n_symptoms, dtype=np.float32)
        for symptom in symptoms:
            if symptom in self.symptom_to_idx:
                vector[self.symptom_to_idx[symptom]] = 1.0
        return vector
    
    def diagnose(self, present_symptoms: List[str], top_k: int = 10) -> List[DiagnosisResult]:
        """
        Простий алгоритм діагностики на основі Jaccard similarity.
        
        Для кожного захворювання обчислює:
        score = |symptoms ∩ disease_symptoms| / |symptoms ∪ disease_symptoms|
        """
        present_set = set(present_symptoms)
        
        if not present_set:
            return []
        
        scores = []
        
        for disease, disease_syms in self.disease_symptoms.items():
            if not disease_syms:
                continue
            
            # Jaccard similarity
            intersection = present_set & disease_syms
            union = present_set | disease_syms
            
            if len(union) > 0:
                jaccard = len(intersection) / len(union)
                
                # Бонус за покриття симптомів пацієнта
                coverage = len(intersection) / len(present_set) if present_set else 0
                
                # Комбінований скор
                score = 0.6 * jaccard + 0.4 * coverage
                
                scores.append(DiagnosisResult(
                    disease_name=disease,
                    probability=score,
                    matching_symptoms=list(intersection)
                ))
        
        # Сортуємо за score
        scores.sort(key=lambda x: x.probability, reverse=True)
        
        # Нормалізуємо top_k до ймовірностей
        top_results = scores[:top_k]
        
        if top_results:
            max_score = top_results[0].probability
            if max_score > 0:
                for r in top_results:
                    r.probability = r.probability / max_score
        
        return top_results


class SOMDiagnosisEngine(SimpleDiagnosisEngine):
    """Діагностичний движок з використанням SOM + Neural Network"""
    
    def __init__(self, db, symptoms_list, symptom_to_idx, som_data, nn_model=None):
        super().__init__(db, symptoms_list, symptom_to_idx)
        self.som_data = som_data
        self.som = som_data.get('som') if som_data else None
        self.unit_to_diseases = som_data.get('unit_to_diseases', {}) if som_data else {}
        self.disease_to_idx = som_data.get('disease_to_idx', {}) if som_data else {}
        self.nn_model = nn_model  # TwoBranchNN instance
    
    def diagnose(self, present_symptoms: List[str], top_k: int = 10) -> List[DiagnosisResult]:
        """Діагностика з SOM + Neural Network"""
        
        # Кодуємо симптоми
        x = self.encode_symptoms(present_symptoms)
        present_set = set(present_symptoms)
        
        # === Етап 1: SOM — отримуємо кандидатів ===
        candidates = set()
        
        if self.som is not None:
            try:
                bmu = self.som.winner(x)
                bmu_key = f"{bmu[0]}_{bmu[1]}"
                
                # BMU та сусіди (5x5 область для більшого покриття)
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        neighbor_key = f"{bmu[0]+di}_{bmu[1]+dj}"
                        if neighbor_key in self.unit_to_diseases:
                            candidates.update(self.unit_to_diseases[neighbor_key])
            except Exception as e:
                pass
        
        # Fallback: якщо кандидатів мало, додаємо всі хвороби
        if len(candidates) < 20:
            candidates = set(self.disease_symptoms.keys())
        
        # === Етап 2: Neural Network — ранжуємо кандидатів ===
        nn_scores = {}
        
        if self.nn_model is not None and self.nn_model.ready:
            try:
                nn_scores = self.nn_model.predict(x)
            except Exception as e:
                pass
        
        # === Етап 3: Комбінуємо SOM + NN + Jaccard ===
        scores = []
        
        for disease in candidates:
            disease_syms = self.disease_symptoms.get(disease, set())
            
            if not disease_syms:
                continue
            
            # Jaccard similarity
            intersection = present_set & disease_syms
            union = present_set | disease_syms
            jaccard = len(intersection) / len(union) if union else 0
            
            # Coverage
            coverage = len(intersection) / len(present_set) if present_set else 0
            
            # NN score
            nn_score = nn_scores.get(disease, 0)
            
            # Комбінований скор
            if nn_score > 0:
                # Якщо є NN — використовуємо його з вагою
                score = 0.5 * nn_score + 0.3 * jaccard + 0.2 * coverage
            else:
                # Без NN — тільки Jaccard
                score = 0.6 * jaccard + 0.4 * coverage
            
            scores.append(DiagnosisResult(
                disease_name=disease,
                probability=score,
                matching_symptoms=list(intersection)
            ))
        
        # Сортуємо
        scores.sort(key=lambda x: x.probability, reverse=True)
        
        # Нормалізуємо
        top_results = scores[:top_k]
        if top_results and top_results[0].probability > 0:
            max_score = top_results[0].probability
            for r in top_results:
                r.probability = r.probability / max_score
        
        return top_results


# ============================================================================
# Ініціалізація Engine
# ============================================================================

@st.cache_resource
def get_engine():
    """Отримати діагностичний движок"""
    db, symptoms_list, symptom_to_idx = load_database()
    
    if not db:
        return None, "Database not found"
    
    # Завантажити SOM
    som_data, som_error = load_som_model()
    
    # Завантажити Neural Network
    nn_checkpoint, nn_error = load_nn_model()
    nn_model = None
    
    if nn_checkpoint:
        try:
            nn_model = TwoBranchNN(nn_checkpoint, len(symptoms_list), len(db))
            if not nn_model.ready:
                nn_model = None
        except Exception as e:
            nn_model = None
    
    # Створити engine
    status_parts = []
    
    if som_data:
        engine = SOMDiagnosisEngine(db, symptoms_list, symptom_to_idx, som_data, nn_model)
        status_parts.append("SOM ✓")
    else:
        engine = SimpleDiagnosisEngine(db, symptoms_list, symptom_to_idx)
        status_parts.append(f"SOM ✗ ({som_error})")
    
    if nn_model and nn_model.ready:
        status_parts.append("NN ✓")
    else:
        status_parts.append(f"NN ✗ ({nn_error if nn_error else 'not loaded'})")
    
    status = " | ".join(status_parts)
    return engine, status


# ============================================================================
# UI Components
# ============================================================================

def show_home():
    """Головна сторінка"""
    st.title("🏥 Dr.Case")
    st.markdown("### Інтелектуальна система медичної діагностики")
    
    st.markdown("""
    **Dr.Case** — це система диференціальної медичної діагностики, що використовує:
    
    - 🧠 **Self-Organizing Map (SOM)** — для визначення клінічного сценарію
    - 🤖 **Neural Network (MultiLabel)** — для ранжування діагнозів
    - 📊 **Jaccard Similarity** — як базовий алгоритм
    
    ---
    
    ### 🚀 Як почати?
    
    Оберіть **"🔍 Швидка діагностика"** в меню зліва 👈
    """)
    
    st.divider()
    
    # Статистика
    db, symptoms_list, _ = load_database()
    engine, status = get_engine()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🦠 Хвороб", len(db))
    col2.metric("🩺 Симптомів", len(symptoms_list))
    col3.metric("📈 Модель", "SOM + NN" if "NN ✓" in (status or "") else "Jaccard")
    
    # Статус моделей
    if status:
        st.info(f"**Статус моделей:** {status}")
    
    st.divider()
    
    st.warning("""
    ⚠️ **Важливо**
    
    Ця система призначена лише для **освітніх та дослідницьких цілей**.  
    Вона **НЕ ЗАМІНЮЄ** консультацію з кваліфікованим лікарем.
    """)
    
    st.caption("*Розроблено: Oleksii Bychkov, Taras Shevchenko National University of Kyiv*")


def show_quick_diagnosis():
    """Швидка діагностика"""
    st.title("🔍 Швидка діагностика")
    st.markdown("Оберіть симптоми та отримайте список найімовірніших діагнозів.")
    
    # Завантаження
    engine, warning = get_engine()
    
    if engine is None:
        st.error(f"❌ Помилка: {warning}")
        st.info("Перевірте наявність файлу бази даних у папці `data/`")
        return
    
    if warning:
        st.info(f"ℹ️ {warning}")
    
    db, symptoms_list, _ = load_database()
    
    st.divider()
    
    # Вибір симптомів
    selected_symptoms = st.multiselect(
        "Оберіть симптоми:",
        options=symptoms_list,
        max_selections=20,
        placeholder="Почніть вводити назву симптому..."
    )
    
    # Швидкий вибір
    st.markdown("**Часті симптоми:**")
    cols = st.columns(6)
    frequent = ["Fever", "Headache", "Cough", "Fatigue", "Nausea", "Vomiting"]
    
    for i, symptom in enumerate(frequent):
        if symptom in symptoms_list:
            if cols[i].button(f"+ {symptom}", key=f"freq_{symptom}"):
                if symptom not in selected_symptoms:
                    st.session_state.setdefault('added_symptoms', []).append(symptom)
                    st.rerun()
    
    # Додаємо симптоми з session_state
    if 'added_symptoms' in st.session_state:
        for s in st.session_state.added_symptoms:
            if s not in selected_symptoms:
                selected_symptoms.append(s)
        st.session_state.added_symptoms = []
    
    st.divider()
    
    # Діагностика
    if st.button("🔬 Діагностувати", type="primary", disabled=not selected_symptoms):
        with st.spinner("Аналіз симптомів..."):
            results = engine.diagnose(selected_symptoms, top_k=10)
            
            if results:
                st.success(f"✅ Знайдено {len(results)} можливих діагнозів")
                
                st.subheader("📋 Результати")
                
                for i, r in enumerate(results, 1):
                    prob_pct = r.probability * 100
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {r.disease_name}**")
                            st.progress(min(r.probability, 1.0), text=f"{prob_pct:.1f}%")
                        
                        with col2:
                            if r.matching_symptoms:
                                with st.expander("Співпадіння"):
                                    for s in r.matching_symptoms[:5]:
                                        st.write(f"• {s}")
            else:
                st.warning("Не знайдено відповідних діагнозів")
    
    # Показуємо вибрані симптоми
    if selected_symptoms:
        st.divider()
        st.markdown(f"**Вибрано симптомів:** {len(selected_symptoms)}")
        st.write(", ".join(selected_symptoms))
    
    st.divider()
    st.caption("⚠️ Ця система не замінює консультацію з лікарем.")


def show_database():
    """База даних"""
    st.title("📊 База даних")
    
    db, symptoms_list, _ = load_database()
    
    tab1, tab2 = st.tabs(["🦠 Хвороби", "🩺 Симптоми"])
    
    with tab1:
        st.subheader(f"Всього: {len(db)} захворювань")
        
        search = st.text_input("🔍 Пошук:", key="disease_search")
        
        diseases = list(db.keys())
        if search:
            diseases = [d for d in diseases if search.lower() in d.lower()]
        
        for disease in diseases[:50]:
            with st.expander(disease):
                info = db[disease]
                if isinstance(info, dict) and 'symptoms' in info:
                    symptoms = info['symptoms']
                elif isinstance(info, list):
                    symptoms = info
                else:
                    symptoms = []
                
                st.write(f"**Симптомів:** {len(symptoms)}")
                st.write(", ".join(symptoms[:15]))
                if len(symptoms) > 15:
                    st.write(f"... та ще {len(symptoms) - 15}")
        
        if len(diseases) > 50:
            st.info(f"Показано 50 з {len(diseases)}")
    
    with tab2:
        st.subheader(f"Всього: {len(symptoms_list)} симптомів")
        
        search = st.text_input("🔍 Пошук:", key="symptom_search")
        
        filtered = symptoms_list
        if search:
            filtered = [s for s in symptoms_list if search.lower() in s.lower()]
        
        cols = st.columns(3)
        for i, s in enumerate(filtered[:90]):
            cols[i % 3].write(f"• {s}")
        
        if len(filtered) > 90:
            st.info(f"Показано 90 з {len(filtered)}")


def show_about():
    """Про систему"""
    st.title("ℹ️ Про систему")
    
    st.markdown("""
    ## Dr.Case
    
    **Інтелектуальна система медичної діагностики**
    
    ---
    
    ### 🏗️ Компоненти
    
    | Компонент | Опис |
    |-----------|------|
    | **SOM** | Self-Organizing Map для кластеризації |
    | **Jaccard** | Similarity-based ranking |
    | **База даних** | 844+ захворювань, 460+ симптомів |
    
    ---
    
    ### 👨‍💻 Розробник
    
    **Oleksii Bychkov**  
    Taras Shevchenko National University of Kyiv
    
    ---
    
    ### ⚠️ Застереження
    
    Система призначена **ВИКЛЮЧНО** для освітніх цілей.
    
    ---
    
    *Версія: 1.0.0*
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    # Sidebar
    st.sidebar.title("🏥 Dr.Case")
    
    page = st.sidebar.radio(
        "Навігація",
        ["🏠 Головна", "🔍 Швидка діагностика", "📊 База даних", "ℹ️ Про систему"]
    )
    
    if page == "🏠 Головна":
        show_home()
    elif page == "🔍 Швидка діагностика":
        show_quick_diagnosis()
    elif page == "📊 База даних":
        show_database()
    elif page == "ℹ️ Про систему":
        show_about()


if __name__ == "__main__":
    main()
