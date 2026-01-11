"""
Dr.Case API — Routes

REST API endpoints для діагностики.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from dr_case.diagnosis_engine import DiagnosisEngine, DiagnosisSession, SessionStatus
from dr_case.schemas import Question, Patient, Gender

from .models import (
    QuickDiagnoseRequest, QuickDiagnoseResponse,
    StartSessionRequest, SessionResponse, SessionWithQuestionsResponse,
    AnswerRequest, BatchAnswerRequest,
    QuestionResponse, DiagnosisHypothesisResponse, DiagnosisResultResponse,
    HealthResponse, ErrorResponse,
    SymptomListResponse, DiseaseListResponse,
    ConfidenceLevelEnum, SessionStatusEnum
)
from .dependencies import (
    get_engine, get_session_store, get_session, app_state, SessionStore
)


# Роутери
router = APIRouter()
diagnosis_router = APIRouter(prefix="/diagnose", tags=["Diagnosis"])
session_router = APIRouter(prefix="/session", tags=["Session"])
info_router = APIRouter(prefix="/info", tags=["Info"])


# === Helper Functions ===

def hypothesis_to_response(h) -> DiagnosisHypothesisResponse:
    """Конвертувати гіпотезу в response"""
    return DiagnosisHypothesisResponse(
        disease_name=h.disease_name,
        confidence=h.confidence,
        confidence_percent=f"{h.confidence:.1%}",
        matching_symptoms=h.matching_symptoms
    )


def question_to_response(q: Question) -> QuestionResponse:
    """Конвертувати питання в response"""
    return QuestionResponse(
        symptom=q.symptom,
        text=q.text,
        information_gain=q.information_gain,
        explanation=q.explanation
    )


def session_to_response(session: DiagnosisSession) -> SessionResponse:
    """Конвертувати сесію в response"""
    return SessionResponse(
        session_id=session.session_id,
        status=SessionStatusEnum(session.status.value),
        current_cycle=session.current_cycle,
        questions_asked=session.n_questions_asked,
        present_symptoms=session.present_symptoms,
        absent_symptoms=session.absent_symptoms,
        candidates_count=session.n_candidates,
        top_diagnosis=session.top_diagnosis.disease_name if session.top_diagnosis else None,
        top_confidence=session.top_confidence,
        confidence_level=ConfidenceLevelEnum(session.confidence_level.value),
        should_continue=session.should_continue
    )


# === Health Check ===

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check"
)
async def health_check():
    """Перевірка здоров'я сервісу"""
    return HealthResponse(
        status="ok" if app_state.is_initialized else "not_initialized",
        version="1.0.0",
        components=app_state.get_health()
    )


# === Diagnosis Endpoints ===

@diagnosis_router.post(
    "/quick",
    response_model=QuickDiagnoseResponse,
    summary="Quick diagnosis",
    description="Швидка діагностика без інтерактивних питань"
)
async def quick_diagnose(
    request: QuickDiagnoseRequest,
    engine: DiagnosisEngine = Depends(get_engine)
):
    """
    Швидка діагностика на основі симптомів.
    
    Повертає список гіпотез з ймовірностями.
    """
    result = engine.diagnose_quick(
        symptoms=request.present_symptoms,
        absent_symptoms=request.absent_symptoms
    )
    
    return QuickDiagnoseResponse(
        hypotheses=[hypothesis_to_response(h) for h in result.hypotheses],
        present_symptoms=result.present_symptoms,
        absent_symptoms=result.absent_symptoms,
        candidates_count=len(result.hypotheses),
        top_diagnosis=result.top_diagnosis.disease_name if result.top_diagnosis else None,
        top_confidence=result.top_confidence,
        confidence_level=ConfidenceLevelEnum(result.confidence_level.value)
    )


# === Session Endpoints ===

@session_router.post(
    "/start",
    response_model=SessionWithQuestionsResponse,
    summary="Start diagnosis session",
    description="Почати інтерактивну сесію діагностики"
)
async def start_session(
    request: StartSessionRequest,
    engine: DiagnosisEngine = Depends(get_engine),
    sessions: SessionStore = Depends(get_session_store)
):
    """
    Почати нову сесію діагностики.
    
    Повертає ID сесії та перші питання.
    """
    # Створюємо Patient якщо є дані
    patient = None
    if request.patient_age or request.patient_gender:
        gender = None
        if request.patient_gender:
            gender = Gender.MALE if request.patient_gender.lower() == "male" else Gender.FEMALE
        patient = Patient(age=request.patient_age, gender=gender)
    
    # Починаємо сесію
    session = engine.start_session(
        initial_symptoms=request.initial_symptoms,
        patient=patient,
        chief_complaint=request.chief_complaint or ""
    )
    
    # Зберігаємо
    sessions.add(session)
    
    # Отримуємо питання
    questions = engine.get_next_questions(session, n=3)
    
    response = session_to_response(session)
    
    return SessionWithQuestionsResponse(
        **response.model_dump(),
        next_questions=[question_to_response(q) for q in questions]
    )


@session_router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session info",
    responses={404: {"model": ErrorResponse}}
)
async def get_session_info(
    session_id: str,
    session: DiagnosisSession = Depends(get_session)
):
    """Отримати інформацію про сесію"""
    return session_to_response(session)


@session_router.get(
    "/{session_id}/questions",
    response_model=List[QuestionResponse],
    summary="Get next questions",
    responses={404: {"model": ErrorResponse}}
)
async def get_questions(
    session_id: str,
    n: int = Query(default=3, ge=1, le=10, description="Кількість питань"),
    session: DiagnosisSession = Depends(get_session),
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Отримати наступні питання"""
    if not session.should_continue:
        return []
    
    questions = engine.get_next_questions(session, n=n)
    return [question_to_response(q) for q in questions]


@session_router.post(
    "/{session_id}/answer",
    response_model=SessionWithQuestionsResponse,
    summary="Answer a question",
    responses={404: {"model": ErrorResponse}}
)
async def answer_question(
    session_id: str,
    request: AnswerRequest,
    session: DiagnosisSession = Depends(get_session),
    engine: DiagnosisEngine = Depends(get_engine)
):
    """
    Відповісти на питання.
    
    Повертає оновлений стан сесії та нові питання.
    """
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Створюємо Question об'єкт
    question = Question(
        symptom=request.symptom,
        text=f"Do you have {request.symptom}?"
    )
    
    # Обробляємо відповідь
    engine.process_answer(session, question, request.answer)
    
    # Отримуємо нові питання
    questions = engine.get_next_questions(session, n=3) if session.should_continue else []
    
    response = session_to_response(session)
    
    return SessionWithQuestionsResponse(
        **response.model_dump(),
        next_questions=[question_to_response(q) for q in questions]
    )


@session_router.post(
    "/{session_id}/answers",
    response_model=SessionWithQuestionsResponse,
    summary="Answer multiple questions",
    responses={404: {"model": ErrorResponse}}
)
async def answer_multiple(
    session_id: str,
    request: BatchAnswerRequest,
    session: DiagnosisSession = Depends(get_session),
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Відповісти на кілька питань одночасно"""
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Обробляємо всі відповіді
    for symptom, answer in request.answers.items():
        question = Question(symptom=symptom, text=f"Do you have {symptom}?")
        engine.process_answer(session, question, answer)
    
    # Отримуємо нові питання
    questions = engine.get_next_questions(session, n=3) if session.should_continue else []
    
    response = session_to_response(session)
    
    return SessionWithQuestionsResponse(
        **response.model_dump(),
        next_questions=[question_to_response(q) for q in questions]
    )


@session_router.post(
    "/{session_id}/complete",
    response_model=DiagnosisResultResponse,
    summary="Complete session",
    responses={404: {"model": ErrorResponse}}
)
async def complete_session(
    session_id: str,
    session: DiagnosisSession = Depends(get_session),
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Завершити сесію та отримати результат"""
    session.complete()
    result = engine.get_result(session)
    explanation = engine.explain_diagnosis(session)
    
    return DiagnosisResultResponse(
        session_id=result.session_id,
        status=SessionStatusEnum.COMPLETED,
        cycles_completed=result.cycles_completed,
        questions_asked=result.questions_asked,
        hypotheses=[hypothesis_to_response(h) for h in result.hypotheses],
        present_symptoms=result.present_symptoms,
        absent_symptoms=result.absent_symptoms,
        top_diagnosis=result.top_diagnosis.disease_name if result.top_diagnosis else None,
        top_confidence=result.top_confidence,
        confidence_level=ConfidenceLevelEnum(result.confidence_level.value),
        is_confident=result.is_confident,
        explanation=explanation
    )


@session_router.delete(
    "/{session_id}",
    summary="Cancel session",
    responses={404: {"model": ErrorResponse}}
)
async def cancel_session(
    session_id: str,
    session: DiagnosisSession = Depends(get_session),
    sessions: SessionStore = Depends(get_session_store)
):
    """Скасувати сесію"""
    session.cancel()
    sessions.remove(session_id)
    return {"message": f"Session {session_id} cancelled"}


# === Info Endpoints ===

@info_router.get(
    "/symptoms",
    response_model=SymptomListResponse,
    summary="List all symptoms"
)
async def list_symptoms(
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Отримати список всіх симптомів"""
    symptoms = engine.encoder.vocabulary.symptoms
    return SymptomListResponse(symptoms=symptoms, count=len(symptoms))


@info_router.get(
    "/diseases",
    response_model=DiseaseListResponse,
    summary="List all diseases"
)
async def list_diseases(
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Отримати список всіх захворювань"""
    diseases = engine.encoder.disease_names
    return DiseaseListResponse(diseases=diseases, count=len(diseases))


@info_router.get(
    "/symptoms/search",
    response_model=SymptomListResponse,
    summary="Search symptoms"
)
async def search_symptoms(
    q: str = Query(..., min_length=2, description="Пошуковий запит"),
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Пошук симптомів за назвою"""
    q_lower = q.lower()
    matching = [s for s in engine.encoder.vocabulary.symptoms if q_lower in s.lower()]
    return SymptomListResponse(symptoms=matching, count=len(matching))


@info_router.get(
    "/diseases/search",
    response_model=DiseaseListResponse,
    summary="Search diseases"
)
async def search_diseases(
    q: str = Query(..., min_length=2, description="Пошуковий запит"),
    engine: DiagnosisEngine = Depends(get_engine)
):
    """Пошук захворювань за назвою"""
    q_lower = q.lower()
    matching = [d for d in engine.encoder.disease_names if q_lower in d.lower()]
    return DiseaseListResponse(diseases=matching, count=len(matching))


# === Include Routers ===

router.include_router(diagnosis_router)
router.include_router(session_router)
router.include_router(info_router)
