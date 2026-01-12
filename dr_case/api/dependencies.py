"""
Dr.Case — API Dependencies

Dependency Injection для FastAPI.
Завантаження моделей, створення контролерів.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import threading
import uuid

from .config import config


class ModelsManager:
    """
    Менеджер моделей — завантажує SOM та NN один раз.
    Singleton pattern.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.is_loaded = False
        self.database = None
        self.symptom_list = []
        self.disease_list = []
        self.controller = None
        self.extractor = None
        self.error = None
    
    def load(self) -> bool:
        """Завантажити всі моделі"""
        if self.is_loaded:
            return True
        
        try:
            print("📦 Завантаження моделей...")
            
            # 1. База даних
            if not config.database_path or not Path(config.database_path).exists():
                self.error = f"Database not found: {config.database_path}"
                return False
            
            with open(config.database_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            
            self.disease_list = list(self.database.keys())
            
            # Збираємо симптоми
            all_symptoms = set()
            for disease_data in self.database.values():
                all_symptoms.update(disease_data.get('symptoms', []))
            self.symptom_list = sorted(list(all_symptoms))
            
            print(f"   ✅ Database: {len(self.disease_list)} diseases, {len(self.symptom_list)} symptoms")
            
            # 2. NLP Extractor
            from dr_case.nlp import SymptomExtractor
            self.extractor = SymptomExtractor(self.symptom_list)
            print("   ✅ NLP Extractor loaded")
            
            # 3. DiagnosisCycleController
            if config.som_path and config.nn_path:
                if Path(config.som_path).exists() and Path(config.nn_path).exists():
                    from dr_case.diagnosis_cycle import DiagnosisCycleController
                    
                    self.controller = DiagnosisCycleController.from_models(
                        database_path=config.database_path,
                        som_path=config.som_path,
                        nn_path=config.nn_path,
                        language=config.default_language
                    )
                    print("   ✅ DiagnosisCycleController loaded")
                else:
                    print(f"   ⚠️ Models not found, running in limited mode")
            
            self.is_loaded = True
            print("📦 Всі моделі завантажено!")
            return True
            
        except Exception as e:
            self.error = str(e)
            print(f"❌ Помилка завантаження: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_controller(self):
        """Отримати контролер"""
        if not self.is_loaded:
            self.load()
        return self.controller
    
    def get_extractor(self):
        """Отримати екстрактор"""
        if not self.is_loaded:
            self.load()
        return self.extractor


class SessionManager:
    """
    Менеджер сесій діагностики.
    Зберігає активні сесії в пам'яті.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.sessions: Dict[str, 'DiagnosisSession'] = {}
        self.lock = threading.Lock()
    
    def create_session(
        self,
        symptoms: list,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
        language: str = "uk"
    ) -> 'DiagnosisSession':
        """Створити нову сесію"""
        session_id = str(uuid.uuid4())[:8]
        
        session = DiagnosisSession(
            session_id=session_id,
            initial_symptoms=symptoms,
            patient_age=patient_age,
            patient_sex=patient_sex,
            language=language
        )
        
        with self.lock:
            # Очистка старих сесій
            self._cleanup_old_sessions()
            
            self.sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional['DiagnosisSession']:
        """Отримати сесію"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Видалити сесію"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
        return False
    
    def get_active_count(self) -> int:
        """Кількість активних сесій"""
        return len(self.sessions)
    
    def _cleanup_old_sessions(self):
        """Видалити застарілі сесії"""
        timeout = timedelta(minutes=config.session_timeout_minutes)
        now = datetime.now()
        
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.updated_at > timeout
        ]
        
        for sid in expired:
            del self.sessions[sid]


class DiagnosisSession:
    """
    Сесія діагностики.
    Обгортка над DiagnosisCycleController для одного пацієнта.
    """
    
    def __init__(
        self,
        session_id: str,
        initial_symptoms: list,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
        language: str = "uk"
    ):
        self.session_id = session_id
        self.initial_symptoms = initial_symptoms
        self.patient_age = patient_age
        self.patient_sex = patient_sex
        self.language = language
        
        self.status = "active"
        self.iteration = 0
        
        self.confirmed_symptoms = list(initial_symptoms)
        self.denied_symptoms = []
        
        self.hypotheses = []
        self.current_question = None
        
        self.stop_reason = None
        self.final_diagnosis = None
        
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Ініціалізуємо контролер
        self._init_controller()
    
    def _init_controller(self):
        """Ініціалізувати контролер"""
        models = ModelsManager()
        controller = models.get_controller()
        
        if controller:
            try:
                controller.start_session(self.initial_symptoms)
                self._update_state(controller)
            except Exception as e:
                print(f"Error initializing session: {e}")
    
    def _update_state(self, controller):
        """Оновити стан з контролера"""
        self.iteration = controller.iteration
        
        # Гіпотези
        hypotheses = controller.get_top_hypotheses(10)
        self.hypotheses = [
            {
                "disease": h[0],
                "probability": h[1],
                "rank": i + 1,
                "change": h[2] if len(h) > 2 else None
            }
            for i, h in enumerate(hypotheses)
        ]
        
        # Питання
        if controller.current_question:
            self.status = "waiting_answer"
            self.current_question = {
                "symptom": controller.current_question,
                "text_uk": self._get_question_text(controller.current_question, "uk"),
                "text_en": self._get_question_text(controller.current_question, "en"),
            }
        else:
            self.current_question = None
        
        # Перевірка завершення
        if controller.is_completed:
            self.status = "completed"
            self.stop_reason = controller.stop_reason
            if self.hypotheses:
                self.final_diagnosis = self.hypotheses[0]["disease"]
        
        self.updated_at = datetime.now()
    
    def _get_question_text(self, symptom: str, language: str) -> str:
        """Згенерувати текст питання"""
        if language == "uk":
            return f"Чи є у вас {symptom.lower().replace('_', ' ')}?"
        else:
            return f"Do you have {symptom.lower().replace('_', ' ')}?"
    
    def answer_question(self, answer: Optional[bool]) -> bool:
        """Обробити відповідь на питання"""
        if self.status != "waiting_answer" or not self.current_question:
            return False
        
        symptom = self.current_question["symptom"]
        
        if answer is True:
            self.confirmed_symptoms.append(symptom)
        elif answer is False:
            self.denied_symptoms.append(symptom)
        # None = не знаю — пропускаємо
        
        # Оновлюємо контролер
        models = ModelsManager()
        controller = models.get_controller()
        
        if controller:
            try:
                controller.answer_question(answer)
                self._update_state(controller)
            except Exception as e:
                print(f"Error answering question: {e}")
                return False
        
        return True
    
    def to_dict(self) -> dict:
        """Конвертувати в словник для API"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "iteration": self.iteration,
            "confirmed_symptoms": self.confirmed_symptoms,
            "denied_symptoms": self.denied_symptoms,
            "hypotheses": self.hypotheses,
            "current_question": self.current_question,
            "stop_reason": self.stop_reason,
            "final_diagnosis": self.final_diagnosis,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# Глобальні менеджери
models_manager = ModelsManager()
session_manager = SessionManager()


# Dependency functions для FastAPI
def get_models() -> ModelsManager:
    """Dependency: отримати менеджер моделей"""
    if not models_manager.is_loaded:
        models_manager.load()
    return models_manager


def get_sessions() -> SessionManager:
    """Dependency: отримати менеджер сесій"""
    return session_manager
