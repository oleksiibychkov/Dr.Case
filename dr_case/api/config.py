"""
Dr.Case — API Configuration

Налаштування FastAPI сервера та шляхи до моделей.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class APIConfig:
    """Конфігурація API сервера"""
    
    # Сервер
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True
    
    # CORS
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list = field(default_factory=lambda: ["*"])
    cors_allow_headers: list = field(default_factory=lambda: ["*"])
    
    # Шляхи до моделей
    database_path: Optional[str] = None
    som_path: Optional[str] = None
    nn_path: Optional[str] = None
    
    # Мова за замовчуванням
    default_language: str = "uk"
    
    # Сесії
    max_sessions: int = 1000
    session_timeout_minutes: int = 60
    
    # API
    api_prefix: str = "/api"
    api_version: str = "v1"
    api_title: str = "Dr.Case API"
    api_description: str = "Інтелектуальна система медичної діагностики"
    
    def __post_init__(self):
        """Автоматичне визначення шляхів"""
        if self.database_path is None or self.som_path is None or self.nn_path is None:
            # Шукаємо project root
            current = Path(__file__).parent.parent.parent
            
            # Пробуємо різні варіанти
            possible_roots = [
                current,
                current.parent,
                Path.cwd(),
            ]
            
            for root in possible_roots:
                db_path = root / "data" / "unified_disease_symptom_merged.json"
                som_path = root / "models" / "som_merged.pkl"
                nn_path = root / "models" / "nn_two_branch.pt"
                
                if db_path.exists():
                    self.database_path = str(db_path)
                    self.som_path = str(som_path)
                    self.nn_path = str(nn_path)
                    break
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Створити конфігурацію з environment variables"""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "true").lower() == "true",
            database_path=os.getenv("DATABASE_PATH"),
            som_path=os.getenv("SOM_PATH"),
            nn_path=os.getenv("NN_PATH"),
            default_language=os.getenv("DEFAULT_LANGUAGE", "uk"),
        )


# Глобальна конфігурація
config = APIConfig()
