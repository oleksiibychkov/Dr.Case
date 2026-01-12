"""
Dr.Case — Sessions Routes

Endpoints для інтерактивних сесій діагностики:
- Створення сесії
- Отримання стану
- Відповідь на питання
- Зворотний зв'язок
- Закриття сесії
"""

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    get_models, get_sessions,
    ModelsManager, SessionManager
)
from ..models import (
    CreateSessionRequest,
    SessionState,
    AnswerRequest,
    AnswerResponse,
    FeedbackRequest,
    FeedbackResponse,
    SessionStatus,
    SessionQuestion,
    Hypothesis,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


def session_to_response(session) -> SessionState:
    """Конвертувати сесію в Pydantic модель"""
    data = session.to_dict()
    
    # Конвертуємо гіпотези
    hypotheses = [
        Hypothesis(
            disease=h["disease"],
            probability=h["probability"],
            rank=h["rank"],
            change=h.get("change")
        )
        for h in data.get("hypotheses", [])
    ]
    
    # Конвертуємо питання
    question = None
    if data.get("current_question"):
        q = data["current_question"]
        question = SessionQuestion(
            symptom=q["symptom"],
            text_uk=q["text_uk"],
            text_en=q["text_en"],
            explanation=q.get("explanation")
        )
    
    return SessionState(
        session_id=data["session_id"],
        status=SessionStatus(data["status"]),
        iteration=data["iteration"],
        confirmed_symptoms=data["confirmed_symptoms"],
        denied_symptoms=data["denied_symptoms"],
        hypotheses=hypotheses,
        current_question=question,
        stop_reason=data.get("stop_reason"),
        final_diagnosis=data.get("final_diagnosis"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.post("", response_model=SessionState)
async def create_session(
    request: CreateSessionRequest,
    models: ModelsManager = Depends(get_models),
    sessions: SessionManager = Depends(get_sessions)
) -> SessionState:
    """
    Створити нову сесію діагностики.
    
    Можна передати:
    - **symptoms**: Список симптомів
    - **text**: Текст з описом скарг (буде оброблено NLP)
    
    Приклад:
    ```json
    {
        "symptoms": ["Headache", "Fever"],
        "language": "uk"
    }
    ```
    
    або:
    
    ```json
    {
        "text": "Болить голова і температура 38",
        "language": "uk"
    }
    ```
    """
    # Отримуємо симптоми
    symptoms = []
    
    if request.symptoms:
        symptoms = request.symptoms
    elif request.text:
        if not models.extractor:
            raise HTTPException(
                status_code=503,
                detail="NLP extractor not available"
            )
        
        extraction = models.extractor.extract(request.text)
        symptoms = extraction.symptoms
    
    if not symptoms:
        raise HTTPException(
            status_code=400,
            detail="No symptoms provided or extracted"
        )
    
    # Створюємо сесію
    session = sessions.create_session(
        symptoms=symptoms,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
        language=request.language.value
    )
    
    return session_to_response(session)


@router.get("/{session_id}", response_model=SessionState)
async def get_session(
    session_id: str,
    sessions: SessionManager = Depends(get_sessions)
) -> SessionState:
    """
    Отримати поточний стан сесії.
    
    Повертає:
    - Статус сесії
    - Підтверджені/заперечені симптоми
    - Поточні гіпотези
    - Поточне питання (якщо є)
    """
    session = sessions.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    return session_to_response(session)


@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def answer_question(
    session_id: str,
    request: AnswerRequest,
    sessions: SessionManager = Depends(get_sessions)
) -> AnswerResponse:
    """
    Відповісти на поточне питання.
    
    - **answer**: true = так, false = ні, null = не знаю
    
    Приклад:
    ```json
    {
        "answer": true
    }
    ```
    """
    session = sessions.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    if session.status != "waiting_answer":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not waiting for answer (status: {session.status})"
        )
    
    success = session.answer_question(request.answer)
    
    return AnswerResponse(
        accepted=success,
        session_state=session_to_response(session)
    )


@router.post("/{session_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    session_id: str,
    request: FeedbackRequest,
    sessions: SessionManager = Depends(get_sessions)
) -> FeedbackResponse:
    """
    Надіслати зворотний зв'язок про результат лікування.
    
    Типи feedback:
    - **treatment_success**: Лікування допомогло
    - **treatment_failed**: Лікування не допомогло
    - **new_symptom**: З'явився новий симптом
    - **condition_changed**: Стан змінився
    
    Приклад:
    ```json
    {
        "feedback_type": "treatment_failed",
        "diagnosis": "Flu",
        "comment": "Симптоми не зменшились після 3 днів"
    }
    ```
    """
    session = sessions.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    action = ""
    
    if request.feedback_type.value == "treatment_success":
        session.status = "completed"
        action = "Session closed successfully"
        
    elif request.feedback_type.value == "treatment_failed":
        # TODO: Перезапустити діагностику з виключенням поточного діагнозу
        action = "Noted for reanalysis"
        
    elif request.feedback_type.value == "new_symptom":
        if request.new_symptom:
            session.confirmed_symptoms.append(request.new_symptom)
            action = f"Added symptom: {request.new_symptom}"
        
    elif request.feedback_type.value == "condition_changed":
        action = "Condition change noted"
    
    return FeedbackResponse(
        accepted=True,
        action_taken=action,
        session_state=session_to_response(session)
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    sessions: SessionManager = Depends(get_sessions)
) -> dict:
    """
    Закрити та видалити сесію.
    """
    success = sessions.delete_session(session_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    return {"deleted": True, "session_id": session_id}


@router.get("")
async def list_sessions(
    sessions: SessionManager = Depends(get_sessions)
) -> dict:
    """
    Отримати список активних сесій (для адміністрування).
    """
    return {
        "active_sessions": sessions.get_active_count(),
        "session_ids": list(sessions.sessions.keys())
    }
