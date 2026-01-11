"""
Dr.Case API — FastAPI Application

Головний файл REST API для діагностичної системи.

Запуск:
    uvicorn dr_case.api.main:app --reload --port 8000
    
Або:
    python -m dr_case.api.main
    
Документація:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .routes import router
from .dependencies import app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events"""
    # Startup
    print("=" * 50)
    print("Dr.Case API — Starting...")
    print("=" * 50)
    
    try:
        # Ініціалізуємо engine
        som_path = os.getenv("SOM_MODEL_PATH", "models/som_optimized.pkl")
        nn_path = os.getenv("NN_MODEL_PATH", "models/nn_model.pt")
        db_path = os.getenv("DATABASE_PATH", "data/unified_disease_symptom_data_full.json")
        
        app_state.initialize(
            som_model_path=som_path,
            nn_model_path=nn_path if os.path.exists(nn_path) else None,
            database_path=db_path
        )
        
        print("✓ API ready!")
        print(f"  Swagger UI: http://localhost:8000/docs")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    print("\nDr.Case API — Shutting down...")


# Створюємо FastAPI app
app = FastAPI(
    title="Dr.Case API",
    description="""
# Dr.Case — Медична діагностична система

REST API для диференціальної діагностики на основі симптомів.

## Можливості

- **Швидка діагностика** — миттєвий аналіз симптомів
- **Інтерактивна діагностика** — покрокове уточнення через питання
- **842 захворювання** та **461 симптом** у базі даних

## Як використовувати

### 1. Швидка діагностика
```
POST /diagnose/quick
{
    "present_symptoms": ["fever", "cough", "headache"]
}
```

### 2. Інтерактивна сесія
```
# Почати сесію
POST /session/start
{
    "initial_symptoms": ["fever", "cough"]
}

# Відповісти на питання
POST /session/{id}/answer
{
    "symptom": "headache",
    "answer": true
}

# Завершити та отримати результат
POST /session/{id}/complete
```

## Технології

- **SOM (Self-Organizing Map)** — кластеризація діагнозів
- **Neural Network** — ранжування кандидатів
- **Information Gain** — вибір оптимальних питань
    """,
    version="1.0.0",
    contact={
        "name": "Dr.Case Team",
        "url": "https://github.com/oleksiibychkov/Dr.Case",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production обмежте домени
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Підключаємо роути
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Кореневий endpoint"""
    return {
        "name": "Dr.Case API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Для запуску напряму
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "dr_case.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
