"""
Dr.Case — Diagnosis Routes

Endpoints для швидкої діагностики та сесій.
"""

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import (
    get_models, get_sessions,
    ModelsManager, SessionManager
)
from ..models import (
    QuickDiagnoseRequest,
    QuickDiagnoseResponse,
    Hypothesis,
    CreateSessionRequest,
    SessionState,
    AnswerRequest,
    AnswerResponse,
    FeedbackRequest,
    FeedbackResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/diagnose", tags=["Diagnosis"])


@router.post("", response_model=QuickDiagnoseResponse)
async def quick_diagnose(
    request: QuickDiagnoseRequest,
    models: ModelsManager = Depends(get_models)
) -> QuickDiagnoseResponse:
    """
    Швидка діагностика за списком симптомів.
    
    Повертає топ-K найімовірніших діагнозів без інтерактивного уточнення.
    
    Приклад:
    ```json
    {
        "symptoms": ["Headache", "Fever", "Cough"],
        "top_k": 10
    }
    ```
    """
    start_time = time.time()
    
    if not models.controller:
        raise HTTPException(
            status_code=503,
            detail="Diagnosis controller not available"
        )
    
    # Валідація симптомів
    valid_symptoms = []
    for symptom in request.symptoms:
        # Пробуємо знайти в базі
        matched = None
        for s in models.symptom_list:
            if s.lower() == symptom.lower():
                matched = s
                break
        
        if matched:
            valid_symptoms.append(matched)
        else:
            # Пробуємо через NLP
            if models.extractor:
                result = models.extractor.match_single(symptom)
                if result:
                    valid_symptoms.append(result.symptom)
    
    if not valid_symptoms:
        raise HTTPException(
            status_code=400,
            detail="No valid symptoms found. Check symptom names."
        )
    
    # Запускаємо діагностику
    try:
        models.controller.start_session(valid_symptoms)
        hypotheses_raw = models.controller.get_top_hypotheses(request.top_k)
        
        hypotheses = [
            Hypothesis(
                disease=h[0],
                probability=h[1],
                rank=i + 1
            )
            for i, h in enumerate(hypotheses_raw)
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis error: {str(e)}"
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return QuickDiagnoseResponse(
        symptoms=valid_symptoms,
        hypotheses=hypotheses,
        processing_time_ms=elapsed_ms
    )


@router.post("/text", response_model=QuickDiagnoseResponse)
async def diagnose_from_text(
    text: str = Query(..., min_length=3, max_length=5000),
    top_k: int = Query(default=10, ge=1, le=50),
    models: ModelsManager = Depends(get_models)
) -> QuickDiagnoseResponse:
    """
    Швидка діагностика з тексту.
    
    Спочатку витягує симптоми з тексту через NLP,
    потім виконує діагностику.
    
    Приклад:
    ```
    /api/diagnose/text?text=Болить голова і температура 38
    ```
    """
    start_time = time.time()
    
    if not models.extractor or not models.controller:
        raise HTTPException(
            status_code=503,
            detail="Models not available"
        )
    
    # Витягуємо симптоми
    extraction = models.extractor.extract(text)
    
    if not extraction.symptoms:
        raise HTTPException(
            status_code=400,
            detail="No symptoms found in text"
        )
    
    # Діагностика
    try:
        models.controller.start_session(extraction.symptoms)
        hypotheses_raw = models.controller.get_top_hypotheses(top_k)
        
        hypotheses = [
            Hypothesis(
                disease=h[0],
                probability=h[1],
                rank=i + 1
            )
            for i, h in enumerate(hypotheses_raw)
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis error: {str(e)}"
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return QuickDiagnoseResponse(
        symptoms=extraction.symptoms,
        hypotheses=hypotheses,
        processing_time_ms=elapsed_ms
    )
