"""
Dr.Case API — Dependencies

Управління залежностями та станом додатку.
"""

from typing import Dict, Optional
from pathlib import Path
import os

from dr_case.diagnosis_engine import DiagnosisEngine, DiagnosisSession


class SessionStore:
    """
    Зберігання активних сесій.
    
    В production використовуйте Redis або базу даних.
    """
    
    def __init__(self):
        self._sessions: Dict[str, DiagnosisSession] = {}
    
    def add(self, session: DiagnosisSession) -> str:
        """Додати сесію"""
        self._sessions[session.session_id] = session
        return session.session_id
    
    def get(self, session_id: str) -> Optional[DiagnosisSession]:
        """Отримати сесію"""
        return self._sessions.get(session_id)
    
    def remove(self, session_id: str) -> bool:
        """Видалити сесію"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def exists(self, session_id: str) -> bool:
        """Перевірити чи існує сесія"""
        return session_id in self._sessions
    
    @property
    def count(self) -> int:
        """Кількість активних сесій"""
        return len(self._sessions)
    
    def cleanup_completed(self) -> int:
        """Видалити завершені сесії"""
        from dr_case.diagnosis_engine import SessionStatus
        
        to_remove = [
            sid for sid, session in self._sessions.items()
            if session.status != SessionStatus.ACTIVE
        ]
        
        for sid in to_remove:
            del self._sessions[sid]
        
        return len(to_remove)


class AppState:
    """
    Глобальний стан додатку.
    
    Містить DiagnosisEngine та SessionStore.
    """
    
    def __init__(self):
        self.engine: Optional[DiagnosisEngine] = None
        self.sessions: SessionStore = SessionStore()
        self._initialized: bool = False
    
    def initialize(
        self,
        som_model_path: Optional[str] = None,
        nn_model_path: Optional[str] = None,
        database_path: Optional[str] = None
    ) -> None:
        """
        Ініціалізувати engine.
        
        Args:
            som_model_path: Шлях до SOM моделі
            nn_model_path: Шлях до NN моделі
            database_path: Шлях до бази даних
        """
        # Визначаємо шляхи
        base_dir = Path(os.getcwd())
        
        if som_model_path is None:
            som_model_path = str(base_dir / "models" / "som_optimized.pkl")
        
        if nn_model_path is None:
            nn_path = base_dir / "models" / "nn_model.pt"
            nn_model_path = str(nn_path) if nn_path.exists() else None
        
        if database_path is None:
            database_path = str(base_dir / "data" / "unified_disease_symptom_data_full.json")
        
        # Перевіряємо файли
        if not Path(som_model_path).exists():
            raise FileNotFoundError(f"SOM model not found: {som_model_path}")
        
        if not Path(database_path).exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        # Створюємо engine
        self.engine = DiagnosisEngine.from_models(
            som_model_path=som_model_path,
            nn_model_path=nn_model_path,
            database_path=database_path
        )
        
        self._initialized = True
        
        print(f"✓ App initialized")
        print(f"  Diseases: {self.engine.n_diseases}")
        print(f"  Symptoms: {self.engine.n_symptoms}")
        print(f"  Has NN: {self.engine.nn_trainer is not None}")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def has_nn(self) -> bool:
        return self.engine is not None and self.engine.nn_trainer is not None
    
    def get_health(self) -> Dict[str, bool]:
        """Перевірка здоров'я компонентів"""
        return {
            "engine": self.engine is not None,
            "som_model": self.engine is not None,
            "nn_model": self.has_nn,
            "sessions_store": True,
        }


# Глобальний стан
app_state = AppState()


def get_engine() -> DiagnosisEngine:
    """Dependency для отримання engine"""
    if not app_state.is_initialized or app_state.engine is None:
        raise RuntimeError("App not initialized. Call app_state.initialize() first.")
    return app_state.engine


def get_session_store() -> SessionStore:
    """Dependency для отримання session store"""
    return app_state.sessions


def get_session(session_id: str) -> DiagnosisSession:
    """Отримати сесію або викинути помилку"""
    session = app_state.sessions.get(session_id)
    if session is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session
