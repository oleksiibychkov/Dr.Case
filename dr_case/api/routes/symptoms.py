"""
Dr.Case — Symptoms Routes

Endpoints для роботи з симптомами:
- Список всіх симптомів
- Пошук симптомів
- NLP витягування симптомів з тексту
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_models, ModelsManager
from ..models import (
    SymptomInfo,
    SymptomSearchResponse,
    ExtractSymptomsRequest,
    ExtractSymptomsResponse,
)

router = APIRouter(prefix="/symptoms", tags=["Symptoms"])


@router.get("", response_model=List[SymptomInfo])
async def list_symptoms(
    models: ModelsManager = Depends(get_models),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
) -> List[SymptomInfo]:
    """
    Отримати список всіх симптомів.
    
    - **limit**: Максимальна кількість (1-1000)
    - **offset**: Зсув для пагінації
    """
    symptoms = models.symptom_list[offset:offset + limit]
    
    return [
        SymptomInfo(name=s)
        for s in symptoms
    ]


@router.get("/search", response_model=SymptomSearchResponse)
async def search_symptoms(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    models: ModelsManager = Depends(get_models)
) -> SymptomSearchResponse:
    """
    Пошук симптомів за текстом.
    
    - **q**: Пошуковий запит
    - **limit**: Максимальна кількість результатів
    
    Шукає в назвах симптомів та синонімах.
    """
    query_lower = q.lower()
    
    # Прямий пошук
    results = []
    for symptom in models.symptom_list:
        if query_lower in symptom.lower():
            results.append(SymptomInfo(name=symptom))
            if len(results) >= limit:
                break
    
    # Якщо мало результатів — шукаємо через NLP
    if len(results) < limit and models.extractor:
        extraction = models.extractor.extract(q)
        for symptom in extraction.symptoms:
            if symptom not in [r.name for r in results]:
                results.append(SymptomInfo(name=symptom))
                if len(results) >= limit:
                    break
    
    return SymptomSearchResponse(
        query=q,
        results=results,
        total=len(results)
    )


@router.get("/count")
async def count_symptoms(
    models: ModelsManager = Depends(get_models)
) -> dict:
    """Отримати кількість симптомів в базі"""
    return {
        "total": len(models.symptom_list)
    }


@router.post("/extract", response_model=ExtractSymptomsResponse)
async def extract_symptoms(
    request: ExtractSymptomsRequest,
    models: ModelsManager = Depends(get_models)
) -> ExtractSymptomsResponse:
    """
    Витягнути симптоми з тексту за допомогою NLP.
    
    Підтримує:
    - Українську та англійську мови
    - Нечітке співставлення
    - Витягування вітальних показників (температура, тиск, пульс)
    - Витягування тривалості симптомів
    
    Приклад:
    ```json
    {
        "text": "Болить голова вже 3 дні, температура 38.5"
    }
    ```
    """
    if not models.extractor:
        raise HTTPException(
            status_code=503,
            detail="NLP extractor not available"
        )
    
    result = models.extractor.extract(request.text)
    
    return ExtractSymptomsResponse(
        original_text=result.original_text,
        symptoms=result.symptoms,
        negated_symptoms=result.negated_symptoms,
        vitals=result.vitals.to_dict(),
        duration=result.duration.to_dict(),
        language=result.language.value,
        confidence=result.confidence
    )
