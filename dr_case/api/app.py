"""
Dr.Case — FastAPI Application

Головний файл FastAPI додатку.

Запуск:
    uvicorn dr_case.api.app:app --reload --host 0.0.0.0 --port 8000
    
    або:
    
    python scripts/run_api.py
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from .config import config
from .dependencies import models_manager
from .routes import (
    health_router,
    symptoms_router,
    diagnosis_router,
    sessions_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager — завантаження моделей при старті.
    """
    print("=" * 60)
    print("🏥 Dr.Case API Starting...")
    print("=" * 60)
    
    # Завантажуємо моделі
    success = models_manager.load()
    
    if success:
        print("✅ API ready!")
    else:
        print(f"⚠️ API starting in limited mode: {models_manager.error}")
    
    print("=" * 60)
    print(f"📍 Swagger UI: http://{config.host}:{config.port}/docs")
    print(f"📍 ReDoc: http://{config.host}:{config.port}/redoc")
    print("=" * 60)
    
    yield
    
    # Cleanup при зупинці
    print("🛑 Dr.Case API Stopping...")


# Створюємо додаток
app = FastAPI(
    title=config.api_title,
    description=config.api_description,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_allow_credentials,
    allow_methods=config.cors_allow_methods,
    allow_headers=config.cors_allow_headers,
)


# Middleware для логування запитів
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Логуємо тільки API запити
    if request.url.path.startswith("/api"):
        print(f"📨 {request.method} {request.url.path} → {response.status_code} ({process_time*1000:.1f}ms)")
    
    return response


# Глобальний обробник помилок
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"❌ Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.debug else None
        }
    )


# Підключаємо роутери
app.include_router(health_router)
app.include_router(symptoms_router, prefix="/api")
app.include_router(diagnosis_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")


# Додатковий endpoint для списку хвороб
@app.get("/api/diseases", tags=["Database"])
async def list_diseases(
    limit: int = 100,
    offset: int = 0
):
    """Отримати список хвороб з бази"""
    diseases = models_manager.disease_list[offset:offset + limit]
    return {
        "diseases": diseases,
        "total": len(models_manager.disease_list),
        "limit": limit,
        "offset": offset
    }


@app.get("/api/diseases/{disease_name}", tags=["Database"])
async def get_disease(disease_name: str):
    """Отримати інформацію про хворобу"""
    if not models_manager.database:
        return {"error": "Database not loaded"}
    
    # Шукаємо хворобу (case-insensitive)
    for name, data in models_manager.database.items():
        if name.lower() == disease_name.lower():
            return {
                "name": name,
                "symptoms": data.get("symptoms", []),
                "symptom_count": len(data.get("symptoms", []))
            }
    
    return {"error": f"Disease '{disease_name}' not found"}
