"""
Dr.Case — Головний діагностичний движок

DiagnosisEngine об'єднує всі компоненти:
1. CandidateSelector (SOM) — звуження простору пошуку
2. DiagnosisRanker (NN) — ранжування кандидатів
3. QuestionGenerator — генерація уточнюючих питань

Реалізує циклічний алгоритм діагностики (натхнений House MD):
1. Початкові симптоми → Кандидати → Гіпотези
2. Питання → Нові симптоми → Оновлені гіпотези
3. Repeat until confidence >= threshold
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from dr_case.config import CandidateSelectorConfig
from dr_case.schemas import DiagnosisHypothesis, Question, Patient
from dr_case.encoding import DiseaseEncoder, SymptomVocabulary, PatientEncoder
from dr_case.candidate_selector import CandidateSelector, SelectionResult
from dr_case.question_engine import QuestionGenerator, QuestionStrategy

from .session import DiagnosisSession, SessionStatus, CycleResult, ConfidenceLevel


@dataclass
class DiagnosisResult:
    """Фінальний результат діагностики"""
    # Топ гіпотези
    hypotheses: List[DiagnosisHypothesis]
    
    # Метадані
    session_id: str
    cycles_completed: int
    questions_asked: int
    
    # Симптоми
    present_symptoms: List[str]
    absent_symptoms: List[str]
    
    # Статус
    confidence_level: ConfidenceLevel
    is_confident: bool
    
    @property
    def top_diagnosis(self) -> Optional[DiagnosisHypothesis]:
        return self.hypotheses[0] if self.hypotheses else None
    
    @property
    def top_confidence(self) -> float:
        return self.top_diagnosis.confidence if self.top_diagnosis else 0.0
    
    def get_top_n(self, n: int = 5) -> List[DiagnosisHypothesis]:
        return self.hypotheses[:n]


class DiagnosisEngine:
    """
    Головний діагностичний движок.
    
    Приклад використання:
        # Ініціалізація
        engine = DiagnosisEngine.from_models(
            som_model_path="models/som_optimized.pkl",
            nn_model_path="models/nn_model.pt",
            database_path="data/unified_disease_symptom_data_full.json"
        )
        
        # Швидка діагностика (без питань)
        result = engine.diagnose_quick(["fever", "cough", "headache"])
        print(f"Top: {result.top_diagnosis.disease_name}")
        
        # Інтерактивна діагностика
        session = engine.start_session(["fever", "cough"])
        
        while session.should_continue:
            # Отримуємо питання
            question = engine.get_next_question(session)
            
            # Імітуємо відповідь користувача
            answer = get_user_answer(question)
            
            # Оновлюємо сесію
            engine.process_answer(session, question, answer)
        
        result = engine.get_result(session)
    """
    
    def __init__(
        self,
        candidate_selector: CandidateSelector,
        question_generator: QuestionGenerator,
        disease_encoder: DiseaseEncoder,
        nn_trainer = None,  # Optional NNTrainer
        confidence_threshold: float = 0.8,
        max_cycles: int = 5,
        max_questions_per_cycle: int = 3
    ):
        """
        Args:
            candidate_selector: Selector для звуження кандидатів
            question_generator: Генератор питань
            disease_encoder: Encoder для симптомів
            nn_trainer: Навчений NNTrainer (опціонально)
            confidence_threshold: Поріг впевненості для зупинки
            max_cycles: Максимум циклів діагностики
            max_questions_per_cycle: Максимум питань за цикл
        """
        self.selector = candidate_selector
        self.question_gen = question_generator
        self.encoder = disease_encoder
        self.nn_trainer = nn_trainer
        
        self.confidence_threshold = confidence_threshold
        self.max_cycles = max_cycles
        self.max_questions_per_cycle = max_questions_per_cycle
        
        # Patient encoder
        self.patient_encoder = PatientEncoder(disease_encoder.vocabulary)
    
    @classmethod
    def from_models(
        cls,
        som_model_path: str,
        database_path: str,
        nn_model_path: Optional[str] = None,
        **kwargs
    ) -> "DiagnosisEngine":
        """
        Створити з файлів моделей.
        
        Args:
            som_model_path: Шлях до SOM моделі (.pkl)
            database_path: Шлях до бази даних
            nn_model_path: Шлях до NN моделі (.pt), опціонально
            **kwargs: Додаткові параметри
            
        Returns:
            DiagnosisEngine
        """
        # CandidateSelector
        selector = CandidateSelector.from_model_file(
            som_model_path,
            database_path
        )
        
        # QuestionGenerator
        question_gen = QuestionGenerator.from_database(
            database_path,
            strategy=QuestionStrategy.HYBRID
        )
        
        # DiseaseEncoder
        encoder = DiseaseEncoder.from_database(database_path)
        
        # NNTrainer (якщо є)
        nn_trainer = None
        if nn_model_path and Path(nn_model_path).exists():
            from dr_case.neural_network import NNTrainer
            nn_trainer = NNTrainer.load(nn_model_path)
        
        return cls(
            candidate_selector=selector,
            question_generator=question_gen,
            disease_encoder=encoder,
            nn_trainer=nn_trainer,
            **kwargs
        )
    
    @classmethod
    def from_database(
        cls,
        database_path: str,
        train_som: bool = True,
        som_epochs: int = 500,
        **kwargs
    ) -> "DiagnosisEngine":
        """
        Створити та навчити з бази даних.
        
        Args:
            database_path: Шлях до бази даних
            train_som: Чи навчати SOM
            som_epochs: Кількість епох SOM
            **kwargs: Додаткові параметри
            
        Returns:
            DiagnosisEngine
        """
        # DiseaseEncoder
        encoder = DiseaseEncoder.from_database(database_path)
        
        # CandidateSelector
        selector = CandidateSelector.from_database(
            database_path,
            epochs=som_epochs if train_som else 100
        )
        
        # QuestionGenerator
        question_gen = QuestionGenerator.from_encoder(encoder)
        
        return cls(
            candidate_selector=selector,
            question_generator=question_gen,
            disease_encoder=encoder,
            **kwargs
        )
    
    def start_session(
        self,
        initial_symptoms: List[str],
        patient: Optional[Patient] = None,
        chief_complaint: str = ""
    ) -> DiagnosisSession:
        """
        Почати нову сесію діагностики.
        
        Args:
            initial_symptoms: Початкові симптоми
            patient: Інформація про пацієнта
            chief_complaint: Головна скарга
            
        Returns:
            DiagnosisSession
        """
        session = DiagnosisSession(
            patient=patient,
            chief_complaint=chief_complaint,
            max_cycles=self.max_cycles,
            max_questions_per_cycle=self.max_questions_per_cycle,
            confidence_threshold=self.confidence_threshold
        )
        
        # Додаємо початкові симптоми
        session.add_symptoms(initial_symptoms, present=True)
        
        # Перший прохід
        self._update_diagnosis(session)
        
        return session
    
    def _update_diagnosis(self, session: DiagnosisSession) -> None:
        """Оновити діагноз на основі поточних симптомів"""
        # 1. Отримуємо кандидатів через SOM
        selection = self.selector.select(
            present_symptoms=session.present_symptoms,
            absent_symptoms=session.absent_symptoms
        )
        
        # 2. Ранжуємо кандидатів
        if self.nn_trainer and selection.candidates:
            # Використовуємо Neural Network
            from dr_case.neural_network import DiagnosisRanker
            ranker = DiagnosisRanker(self.nn_trainer, self.encoder.vocabulary)
            
            ranking = ranker.rank(
                present_symptoms=session.present_symptoms,
                candidates=selection.candidates,
                absent_symptoms=session.absent_symptoms
            )
            
            # Формуємо гіпотези
            hypotheses = []
            for disease, score, prob in ranking.get_top_n(10):
                hypothesis = DiagnosisHypothesis(
                    disease_name=disease,
                    confidence=prob,
                    matching_symptoms=session.present_symptoms.copy()
                )
                hypotheses.append(hypothesis)
            
            scores = {d: s for d, s in zip(ranking.ranked_diseases, ranking.scores)}
            
        else:
            # Простий метод: рівномірні ймовірності
            n_candidates = len(selection.candidates)
            prob = 1.0 / n_candidates if n_candidates > 0 else 0.0
            
            hypotheses = []
            for disease in selection.candidates[:10]:
                hypothesis = DiagnosisHypothesis(
                    disease_name=disease,
                    confidence=prob,
                    matching_symptoms=session.present_symptoms.copy()
                )
                hypotheses.append(hypothesis)
            
            scores = {d: prob for d in selection.candidates}
        
        # 3. Оновлюємо сесію
        session.update_candidates(selection.candidates, scores)
        session.update_hypotheses(hypotheses)
    
    def get_next_question(self, session: DiagnosisSession) -> Optional[Question]:
        """
        Отримати наступне питання.
        
        Args:
            session: Поточна сесія
            
        Returns:
            Question або None якщо питань більше немає
        """
        if not session.should_continue:
            return None
        
        if not session.current_candidates:
            return None
        
        # Генеруємо питання
        question = self.question_gen.generate_single(
            candidates=session.current_candidates,
            known_present=session.present_symptoms,
            known_absent=session.absent_symptoms,
            disease_scores=session.candidate_scores
        )
        
        return question
    
    def get_next_questions(
        self,
        session: DiagnosisSession,
        n: int = 3
    ) -> List[Question]:
        """
        Отримати кілька наступних питань.
        
        Args:
            session: Поточна сесія
            n: Кількість питань
            
        Returns:
            Список Question
        """
        if not session.should_continue:
            return []
        
        if not session.current_candidates:
            return []
        
        questions = self.question_gen.generate(
            candidates=session.current_candidates,
            known_present=session.present_symptoms,
            known_absent=session.absent_symptoms,
            disease_scores=session.candidate_scores,
            n_questions=n
        )
        
        return questions
    
    def process_answer(
        self,
        session: DiagnosisSession,
        question: Question,
        answer: bool
    ) -> None:
        """
        Обробити відповідь на питання.
        
        Args:
            session: Поточна сесія
            question: Питання
            answer: True = так, False = ні
        """
        # Записуємо відповідь
        session.record_answer(question, answer)
        
        # Оновлюємо діагноз
        self._update_diagnosis(session)
    
    def run_cycle(
        self,
        session: DiagnosisSession,
        answer_func: Callable[[Question], bool]
    ) -> CycleResult:
        """
        Запустити один цикл діагностики.
        
        Args:
            session: Поточна сесія
            answer_func: Функція для отримання відповіді
            
        Returns:
            CycleResult
        """
        cycle_num = session.start_new_cycle()
        candidates_before = session.current_candidates.copy()
        questions_asked = []
        
        # Задаємо питання
        for _ in range(self.max_questions_per_cycle):
            if not session.should_continue:
                break
            
            question = self.get_next_question(session)
            if question is None:
                break
            
            # Отримуємо відповідь
            answer = answer_func(question)
            
            # Обробляємо
            self.process_answer(session, question, answer)
            
            from .session import QuestionAnswer
            questions_asked.append(QuestionAnswer(question, answer))
        
        # Формуємо результат циклу
        result = CycleResult(
            cycle_number=cycle_num,
            candidates_before=candidates_before,
            top_hypotheses=session.top_hypotheses.copy(),
            questions_asked=questions_asked
        )
        
        session.complete_cycle(result)
        
        return result
    
    def run_full_diagnosis(
        self,
        initial_symptoms: List[str],
        answer_func: Callable[[Question], bool],
        patient: Optional[Patient] = None
    ) -> DiagnosisResult:
        """
        Запустити повну діагностику.
        
        Args:
            initial_symptoms: Початкові симптоми
            answer_func: Функція для отримання відповідей
            patient: Інформація про пацієнта
            
        Returns:
            DiagnosisResult
        """
        session = self.start_session(initial_symptoms, patient)
        
        while session.should_continue:
            self.run_cycle(session, answer_func)
        
        session.complete()
        
        return self.get_result(session)
    
    def diagnose_quick(
        self,
        symptoms: List[str],
        absent_symptoms: Optional[List[str]] = None
    ) -> DiagnosisResult:
        """
        Швидка діагностика без питань.
        
        Args:
            symptoms: Присутні симптоми
            absent_symptoms: Відсутні симптоми
            
        Returns:
            DiagnosisResult
        """
        session = DiagnosisSession()
        session.add_symptoms(symptoms, present=True)
        if absent_symptoms:
            session.add_symptoms(absent_symptoms, present=False)
        
        self._update_diagnosis(session)
        session.complete()
        
        return self.get_result(session)
    
    def get_result(self, session: DiagnosisSession) -> DiagnosisResult:
        """
        Отримати результат діагностики.
        
        Args:
            session: Сесія
            
        Returns:
            DiagnosisResult
        """
        return DiagnosisResult(
            hypotheses=session.top_hypotheses.copy(),
            session_id=session.session_id,
            cycles_completed=session.current_cycle,
            questions_asked=session.n_questions_asked,
            present_symptoms=session.present_symptoms.copy(),
            absent_symptoms=session.absent_symptoms.copy(),
            confidence_level=session.confidence_level,
            is_confident=session.top_confidence >= self.confidence_threshold
        )
    
    def explain_diagnosis(
        self,
        session: DiagnosisSession,
        disease: Optional[str] = None
    ) -> str:
        """
        Пояснити діагноз.
        
        Args:
            session: Сесія
            disease: Діагноз для пояснення (або топ)
            
        Returns:
            Текстове пояснення
        """
        if disease is None:
            if session.top_diagnosis:
                disease = session.top_diagnosis.disease_name
            else:
                return "No diagnosis available."
        
        lines = []
        lines.append(f"Diagnosis Explanation: {disease}")
        lines.append("=" * 50)
        
        # Симптоми
        lines.append(f"\nPresent symptoms ({len(session.present_symptoms)}):")
        for s in session.present_symptoms:
            lines.append(f"  ✓ {s}")
        
        if session.absent_symptoms:
            lines.append(f"\nAbsent symptoms ({len(session.absent_symptoms)}):")
            for s in session.absent_symptoms:
                lines.append(f"  ✗ {s}")
        
        # Топ гіпотези
        lines.append(f"\nTop hypotheses:")
        for i, h in enumerate(session.top_hypotheses[:5]):
            marker = "→" if h.disease_name == disease else " "
            lines.append(f"  {marker} {i+1}. {h.disease_name}: {h.confidence:.1%}")
        
        # Цикли
        if session.cycle_history:
            lines.append(f"\nDiagnosis cycles: {len(session.cycle_history)}")
            for cycle in session.cycle_history:
                lines.append(f"  Cycle {cycle.cycle_number}: {len(cycle.questions_asked)} questions")
        
        return "\n".join(lines)
    
    @property
    def n_diseases(self) -> int:
        return len(self.encoder.disease_names)
    
    @property
    def n_symptoms(self) -> int:
        return self.encoder.vector_dim
    
    def __repr__(self) -> str:
        return (
            f"DiagnosisEngine("
            f"diseases={self.n_diseases}, "
            f"symptoms={self.n_symptoms}, "
            f"has_nn={self.nn_trainer is not None}"
            f")"
        )
