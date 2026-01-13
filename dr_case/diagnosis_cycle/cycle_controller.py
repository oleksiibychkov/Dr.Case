"""
Dr.Case — Головний контролер циклічного процесу діагностики

ОНОВЛЕНО для нової архітектури з BMU координатами:
- NN приймає symptoms [460] + BMU coords [2]
- Гіпотези розділяються на "з кластера" та "не з кластера"
- Діагноз "не з кластера" що залишається → потрібні додаткові дослідження

Інтегрує всі компоненти:
- SOM + CandidateSelector
- TwoBranchNN_BMU
- Question Engine
- Stopping Criteria
- Hypothesis Tracker
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from .stopping_criteria import StoppingCriteria, StoppingConfig, StopReason, StopDecision
from .hypothesis_tracker import HypothesisTracker


class SessionPhase(Enum):
    """Фаза сесії"""
    INITIAL = "initial"
    QUESTIONING = "questioning"
    NEED_TESTS = "need_tests"
    AWAITING_RESULTS = "awaiting_results"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DiagnosisQuestion:
    """Питання для пацієнта"""
    symptom: str
    text: str
    explanation: str
    eig: float
    p_yes: float
    p_no: float


@dataclass
class HypothesisInfo:
    """Інформація про гіпотезу"""
    name: str
    probability: float
    in_cluster: bool  # Чи знаходиться в кластері SOM
    
    @property
    def needs_attention(self) -> bool:
        """Потребує уваги якщо висока ймовірність але не з кластера"""
        return not self.in_cluster and self.probability > 0.1


@dataclass
class IterationResult:
    """Результат однієї ітерації"""
    iteration: int
    
    # Вектор симптомів
    symptom_vector: np.ndarray
    
    # BMU координати
    bmu_coords: Tuple[int, int]  # (row, col)
    bmu_coords_normalized: Tuple[float, float]  # (row/H, col/W)
    
    # Кандидати з кластера
    cluster_candidates: List[str]
    
    # Гіпотези від NN
    hypotheses: Dict[str, float]
    
    # Розділені гіпотези
    hypotheses_in_cluster: List[HypothesisInfo]
    hypotheses_outside_cluster: List[HypothesisInfo]
    
    # Питання (якщо є)
    question: Optional[DiagnosisQuestion] = None
    
    # Критерій зупинки
    stop_decision: Optional[StopDecision] = None
    
    @property
    def has_outside_cluster_attention(self) -> bool:
        """Чи є діагнози не з кластера що потребують уваги"""
        return any(h.needs_attention for h in self.hypotheses_outside_cluster)


@dataclass
class DiagnosisCycleResult:
    """Фінальний результат діагностики"""
    # Топ гіпотези
    top_hypotheses: List[Tuple[str, float]]
    
    # Розділені результати
    top_in_cluster: List[Tuple[str, float]]
    top_outside_cluster: List[Tuple[str, float]]
    
    # Причина зупинки
    stop_reason: StopReason
    stop_message: str
    
    # Статистика
    iterations: int
    questions_asked: int
    duration_seconds: float
    
    # Симптоми
    present_symptoms: List[str]
    absent_symptoms: List[str]
    unknown_symptoms: List[str]
    
    # Рекомендації
    recommended_tests: List[str] = field(default_factory=list)
    
    # НОВЕ: Потрібна увага до діагнозів не з кластера
    needs_additional_investigation: bool = False
    outside_cluster_warning: str = ""
    
    # Історія ітерацій
    iteration_history: List[IterationResult] = field(default_factory=list)
    
    @property
    def top_diagnosis(self) -> Optional[str]:
        return self.top_hypotheses[0][0] if self.top_hypotheses else None
    
    @property
    def top_confidence(self) -> float:
        return self.top_hypotheses[0][1] if self.top_hypotheses else 0.0
    
    @property
    def is_confident(self) -> bool:
        return self.top_confidence >= 0.80


class DiagnosisCycleController:
    """
    Головний контролер циклічного процесу діагностики.
    
    НОВА АРХІТЕКТУРА:
    - NN приймає BMU координати [2] замість top-k membership [10]
    - Розділення гіпотез на "з кластера" / "не з кластера"
    - Діагноз "не з кластера" що залишається → додаткові дослідження
    
    Приклад використання:
        controller = DiagnosisCycleController.from_models(
            database_path="data/unified_disease_symptom_merged.json",
            som_path="models/som_merged.pkl",
            nn_path="models/nn_two_branch_bmu.pt"
        )
        
        result = controller.run_full_diagnosis(
            initial_symptoms=['Fever', 'Cough', 'Headache'],
            answer_func=answer_func
        )
        
        # Перевіряємо чи потрібні додаткові дослідження
        if result.needs_additional_investigation:
            print(f"УВАГА: {result.outside_cluster_warning}")
    """
    
    def __init__(
        self,
        database: Dict[str, Any],
        som_model: Any,
        nn_model: Any,
        symptom_vocab: Any,
        som_shape: Tuple[int, int],  # НОВЕ: розміри SOM для нормалізації
        unit_to_diseases: Dict,  # НОВЕ: маппінг юніт → діагнози
        stopping_config: Optional[StoppingConfig] = None,
        language: str = "uk"
    ):
        self.database = database
        self.som_model = som_model
        self.nn_model = nn_model
        self.symptom_vocab = symptom_vocab
        self.som_shape = som_shape  # (H, W)
        self.unit_to_diseases = unit_to_diseases
        self.language = language
        
        # Критерії зупинки
        self.stopping_criteria = StoppingCriteria(
            stopping_config or StoppingConfig()
        )
        
        # Ініціалізуємо Question Engine
        self._init_question_engine()
        
        # Поточна сесія
        self._reset_session()
    
    def _init_question_engine(self):
        """Ініціалізація Question Engine"""
        from dr_case.question_engine import (
            InformationGainCalculator,
            QuestionSelector,
            QuestionGenerator,
            AnswerProcessor,
            SessionState,
            AnswerType
        )
        
        self._AnswerType = AnswerType
        
        # Калькулятор EIG
        eig_calculator = InformationGainCalculator.from_database(self.database)
        
        # Селектор питань
        self.question_selector = QuestionSelector.from_loaded_database(
            database=self.database,
            language=self.language
        )
        
        # Процесор відповідей
        self.answer_processor = AnswerProcessor(eig_calculator)
        
        # Генератор питань
        self.question_generator = QuestionGenerator(language=self.language)
    
    def _reset_session(self):
        """Скидання сесії"""
        from dr_case.question_engine import SessionState
        
        self.session_state = SessionState()
        self.hypothesis_tracker = HypothesisTracker()
        self._phase = SessionPhase.INITIAL
        self.start_time = None
        self.iteration_count = 0
        self.questions_count = 0
        self.current_question = None
        self.iteration_history = []
        
        # НОВЕ: Трекінг діагнозів не з кластера
        self.outside_cluster_tracker: Dict[str, int] = {}  # діагноз → кількість ітерацій
    
    @classmethod
    def from_models(
        cls,
        database_path: str,
        som_path: str,
        nn_path: str,
        stopping_config: Optional[StoppingConfig] = None,
        language: str = "uk"
    ) -> "DiagnosisCycleController":
        """
        Створити контролер з файлів моделей.
        
        Args:
            database_path: Шлях до JSON бази даних
            som_path: Шлях до SOM моделі (.pkl)
            nn_path: Шлях до NN моделі (.pt) - НОВА МОДЕЛЬ nn_two_branch_bmu.pt
            stopping_config: Конфігурація зупинки
            language: Мова (uk/en)
        """
        import torch
        from dr_case.som import SOMModel
        
        # Завантажуємо базу даних
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        # Завантажуємо SOM
        som_model = SOMModel.load(som_path)
        som_shape = som_model._som._weights.shape[:2]  # (H, W)
        
        # Отримуємо unit_to_diseases
        unit_to_diseases = som_model._unit_to_diseases
        
        print(f"   SOM shape: {som_shape}")
        print(f"   Units with diseases: {len(unit_to_diseases)}")
        
        # Завантажуємо NN
        nn_checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
        
        print(f"   NN checkpoint keys: {list(nn_checkpoint.keys())}")
        
        model_config = nn_checkpoint.get('model_config', {})
        print(f"   model_config: {model_config}")
        
        # Перевіряємо тип моделі
        som_dim = model_config.get('som_dim', 2)
        
        if som_dim == 2:
            # НОВА модель з BMU координатами
            from dr_case.neural_network.trainer_bmu import TwoBranchNN_BMU
            
            nn_model = TwoBranchNN_BMU(
                n_symptoms=model_config['n_symptoms'],
                n_diseases=model_config['n_diseases'],
                symptom_hidden=model_config.get('symptom_hidden', [256, 128]),
                bmu_hidden=model_config.get('bmu_hidden', [32, 16]),
                combined_hidden=model_config.get('combined_hidden', [128]),
                dropout=model_config.get('dropout', 0.3),
            )
            print("   Using TwoBranchNN_BMU (BMU coordinates)")
        else:
            # Стара модель - fallback
            from dr_case.neural_network.two_branch_model import TwoBranchNN
            
            nn_model = TwoBranchNN(
                n_symptoms=model_config.get('n_symptoms', 460),
                som_dim=som_dim,
                n_diseases=model_config.get('n_diseases', len(database)),
                symptom_hidden=model_config.get('symptom_hidden', [256, 128]),
                som_hidden=model_config.get('som_hidden', [64, 32]),
                combined_hidden=model_config.get('combined_hidden', [128]),
            )
            print("   Using TwoBranchNN (legacy)")
        
        # Завантажуємо ваги
        state_dict = nn_checkpoint.get('model_state')
        if state_dict:
            nn_model.load_state_dict(state_dict)
        nn_model.eval()
        
        # Зберігаємо metadata
        nn_model.disease_names = nn_checkpoint.get('disease_names', list(database.keys()))
        nn_model.symptom_names = nn_checkpoint.get('symptom_names', [])
        nn_model.som_shape = nn_checkpoint.get('som_shape', som_shape)
        
        # Словник симптомів
        from dr_case.encoding import SymptomVocabulary
        symptom_vocab = SymptomVocabulary.from_database(database_path)
        
        return cls(
            database=database,
            som_model=som_model,
            nn_model=nn_model,
            symptom_vocab=symptom_vocab,
            som_shape=som_shape,
            unit_to_diseases=unit_to_diseases,
            stopping_config=stopping_config,
            language=language
        )
    
    def start_session(self, initial_symptoms: List[str]) -> IterationResult:
        """Почати нову сесію діагностики."""
        self._reset_session()
        
        for symptom in initial_symptoms:
            self.session_state.add_symptom_answer(symptom, self._AnswerType.YES)
        
        self._phase = SessionPhase.QUESTIONING
        self.start_time = datetime.now()
        
        return self._run_iteration()
    
    def _run_iteration(self) -> IterationResult:
        """Виконати одну ітерацію діагностики"""
        self.iteration_count += 1
        
        # 1. Отримати вектор симптомів
        symptom_vector = self._get_symptom_vector()
        
        # 2. Проєкція на SOM → BMU координати
        bmu_coords, bmu_normalized = self._get_bmu_coordinates(symptom_vector)
        
        # 3. Отримати кандидатів з кластера
        cluster_candidates = self._get_cluster_candidates(bmu_coords)
        
        # 4. Передбачення NN
        hypotheses = self._predict_hypotheses(symptom_vector, bmu_normalized)
        
        # 5. Розділити гіпотези на "з кластера" / "не з кластера"
        in_cluster, outside_cluster = self._split_hypotheses(hypotheses, cluster_candidates)
        
        # 6. Оновити трекер діагнозів не з кластера
        self._update_outside_cluster_tracker(outside_cluster)
        
        # 7. Оновити hypothesis tracker
        self.hypothesis_tracker.update(hypotheses, self.iteration_count)
        
        # 8. Перевірити критерії зупинки
        stop_decision = self.stopping_criteria.check(
            current_hypotheses=hypotheses,
            iteration=self.iteration_count,
            questions_asked=self.questions_count,
            hypothesis_history=self.hypothesis_tracker.get_hypothesis_history()[:-1] if self.hypothesis_tracker.get_hypothesis_history() else [],
            elapsed_minutes=self._elapsed_minutes(),
            available_questions=self._count_available_questions()
        )
        
        # 9. Якщо продовжуємо — вибрати питання
        question = None
        if stop_decision.should_continue:
            question = self._select_next_question(hypotheses)
            if question:
                self.current_question = question
        
        # Формуємо результат
        result = IterationResult(
            iteration=self.iteration_count,
            symptom_vector=symptom_vector,
            bmu_coords=bmu_coords,
            bmu_coords_normalized=bmu_normalized,
            cluster_candidates=cluster_candidates,
            hypotheses=hypotheses,
            hypotheses_in_cluster=in_cluster,
            hypotheses_outside_cluster=outside_cluster,
            question=question,
            stop_decision=stop_decision
        )
        
        self.iteration_history.append(result)
        
        # Оновлюємо фазу
        if not stop_decision.should_continue:
            if stop_decision.reason == StopReason.NEED_TEST:
                self._phase = SessionPhase.NEED_TESTS
            else:
                self._phase = SessionPhase.COMPLETED
        
        return result
    
    def _get_symptom_vector(self) -> np.ndarray:
        """Отримати бінарний вектор симптомів"""
        vector = np.zeros(self.symptom_vocab.size, dtype=np.float32)
        
        for symptom in self.session_state.known_symptoms:
            idx = self.symptom_vocab.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = 1.0
        
        return vector
    
    def _get_bmu_coordinates(
        self, 
        symptom_vector: np.ndarray
    ) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        Отримати BMU координати для симптомів.
        
        Returns:
            (bmu_coords, bmu_normalized): ((row, col), (row/H, col/W))
        """
        minisom = self.som_model._som
        bmu = minisom.winner(symptom_vector)
        
        H, W = self.som_shape
        bmu_normalized = (bmu[0] / H, bmu[1] / W)
        
        return bmu, bmu_normalized
    
    def _get_cluster_candidates(self, bmu_coords: Tuple[int, int]) -> List[str]:
        """Отримати діагнози-кандидати з кластера BMU"""
        candidates = set()
        
        # Пробуємо різні формати ключів
        if bmu_coords in self.unit_to_diseases:
            candidates.update(self.unit_to_diseases[bmu_coords])
        elif (int(bmu_coords[0]), int(bmu_coords[1])) in self.unit_to_diseases:
            candidates.update(self.unit_to_diseases[(int(bmu_coords[0]), int(bmu_coords[1]))])
        
        # Також додаємо сусідні юніти (розширений кластер)
        H, W = self.som_shape
        radius = self.stopping_criteria.config.cluster_radius
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                ni, nj = bmu_coords[0] + di, bmu_coords[1] + dj
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor = (ni, nj)
                    if neighbor in self.unit_to_diseases:
                        candidates.update(self.unit_to_diseases[neighbor])
                    elif (int(ni), int(nj)) in self.unit_to_diseases:
                        candidates.update(self.unit_to_diseases[(int(ni), int(nj))])
        
        return list(candidates)
    
    def _predict_hypotheses(
        self,
        symptom_vector: np.ndarray,
        bmu_normalized: Tuple[float, float]
    ) -> Dict[str, float]:
        """Передбачення гіпотез через NN з BMU координатами"""
        import torch
        
        # Створюємо вектор симптомів для NN (lowercase!)
        if hasattr(self.nn_model, 'symptom_names') and self.nn_model.symptom_names:
            nn_symptom_names = self.nn_model.symptom_names
            nn_symptom_to_idx = {s: i for i, s in enumerate(nn_symptom_names)}
            nn_vector = np.zeros(len(nn_symptom_names), dtype=np.float32)
            
            for symptom in self.session_state.known_symptoms:
                key = symptom.lower()
                if key in nn_symptom_to_idx:
                    nn_vector[nn_symptom_to_idx[key]] = 1.0
            
            symptoms_tensor = torch.tensor(nn_vector, dtype=torch.float32).unsqueeze(0)
        else:
            symptoms_tensor = torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0)
        
        # BMU координати (нормалізовані)
        bmu_tensor = torch.tensor(
            [bmu_normalized[0], bmu_normalized[1]], 
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Передбачення
        with torch.no_grad():
            outputs = self.nn_model(symptoms_tensor, bmu_tensor)
            probs = torch.softmax(outputs, dim=-1).squeeze().numpy()
        
        # Формуємо результат
        if hasattr(self.nn_model, 'disease_names') and self.nn_model.disease_names:
            disease_names = self.nn_model.disease_names
        else:
            disease_names = list(self.database.keys())
        
        if len(probs) != len(disease_names):
            return {d: 1.0 / len(disease_names) for d in disease_names}
        
        return {
            disease_names[i]: float(probs[i])
            for i in range(len(disease_names))
        }
    
    def _split_hypotheses(
        self,
        hypotheses: Dict[str, float],
        cluster_candidates: List[str]
    ) -> Tuple[List[HypothesisInfo], List[HypothesisInfo]]:
        """
        Розділити гіпотези на "з кластера" та "не з кластера".
        
        Returns:
            (in_cluster, outside_cluster): Списки HypothesisInfo
        """
        cluster_set = set(cluster_candidates)
        
        in_cluster = []
        outside_cluster = []
        
        # Сортуємо за ймовірністю
        sorted_hyp = sorted(hypotheses.items(), key=lambda x: x[1], reverse=True)
        
        for name, prob in sorted_hyp[:20]:  # Топ-20
            info = HypothesisInfo(name=name, probability=prob, in_cluster=(name in cluster_set))
            
            if info.in_cluster:
                in_cluster.append(info)
            else:
                outside_cluster.append(info)
        
        return in_cluster, outside_cluster
    
    def _update_outside_cluster_tracker(self, outside_cluster: List[HypothesisInfo]):
        """Оновити трекер діагнозів не з кластера"""
        # Збільшуємо лічильник для тих що залишаються
        current_outside = {h.name for h in outside_cluster if h.needs_attention}
        
        for name in current_outside:
            self.outside_cluster_tracker[name] = self.outside_cluster_tracker.get(name, 0) + 1
        
        # Видаляємо ті що зникли
        to_remove = [name for name in self.outside_cluster_tracker if name not in current_outside]
        for name in to_remove:
            del self.outside_cluster_tracker[name]
    
    def _select_next_question(
        self,
        hypotheses: Dict[str, float]
    ) -> Optional[DiagnosisQuestion]:
        """Вибрати наступне питання за EIG"""
        question = self.question_selector.select_question(
            disease_probs=hypotheses,
            known_symptoms=self.session_state.known_symptoms,
            asked_symptoms=self.session_state.all_asked_symptoms,
            negated_symptoms=self.session_state.negated_symptoms
        )
        
        if question is None:
            return None
        
        return DiagnosisQuestion(
            symptom=question.symptom,
            text=question.text,
            explanation=f"EIG: {question.eig:.4f}",
            eig=question.eig,
            p_yes=question.p_yes,
            p_no=question.p_no
        )
    
    def _count_available_questions(self) -> int:
        """Підрахувати кількість доступних питань"""
        asked = set(self.session_state.all_asked_symptoms)
        known = set(self.session_state.known_symptoms)
        unknown = set(self.session_state.unknown_symptoms)
        excluded = asked | known | unknown
        
        all_symptoms = self.symptom_vocab.symptoms
        return len(all_symptoms) - len(excluded)
    
    def _elapsed_minutes(self) -> float:
        """Час з початку сесії в хвилинах"""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() / 60.0
    
    def should_continue(self) -> bool:
        """Чи треба продовжувати діагностику"""
        if self._phase != SessionPhase.QUESTIONING:
            return False
        
        if not self.iteration_history:
            return True
        
        last_result = self.iteration_history[-1]
        if last_result.stop_decision:
            return last_result.stop_decision.should_continue
        
        return True
    
    def get_next_question(self) -> Optional[DiagnosisQuestion]:
        """Отримати поточне питання"""
        return self.current_question
    
    def process_answer(self, answer: Optional[bool]) -> IterationResult:
        """
        Обробити відповідь на питання.
        
        Args:
            answer: True=так, False=ні, None=не знаю
        """
        if self.current_question is None:
            raise ValueError("Немає поточного питання")
        
        symptom = self.current_question.symptom
        self.questions_count += 1
        
        # Визначаємо тип відповіді
        if answer is True:
            answer_type = self._AnswerType.YES
        elif answer is False:
            answer_type = self._AnswerType.NO
        else:
            answer_type = self._AnswerType.UNKNOWN
        
        # Оновлюємо стан
        self.session_state.add_symptom_answer(symptom, answer_type)
        
        # Очищуємо поточне питання
        self.current_question = None
        
        # Наступна ітерація
        return self._run_iteration()
    
    def get_result(self) -> DiagnosisCycleResult:
        """Отримати фінальний результат"""
        if not self.iteration_history:
            raise ValueError("Сесія не почата")
        
        last_result = self.iteration_history[-1]
        
        # Топ гіпотези
        sorted_hyp = sorted(
            last_result.hypotheses.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Розділені топ гіпотези
        top_in_cluster = [
            (h.name, h.probability) 
            for h in last_result.hypotheses_in_cluster[:5]
        ]
        top_outside_cluster = [
            (h.name, h.probability) 
            for h in last_result.hypotheses_outside_cluster[:5]
        ]
        
        # Визначаємо причину зупинки
        if last_result.stop_decision:
            stop_reason = last_result.stop_decision.reason
            stop_message = last_result.stop_decision.message
        else:
            stop_reason = StopReason.MAX_ITERATIONS
            stop_message = "Досягнуто максимум ітерацій"
        
        # Час
        duration = 0.0
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        # НОВЕ: Перевіряємо діагнози не з кластера
        needs_investigation = False
        warning = ""
        
        # Якщо діагноз не з кластера залишався 3+ ітерації з ймовірністю > 10%
        persistent_outside = [
            name for name, count in self.outside_cluster_tracker.items()
            if count >= 3
        ]
        
        if persistent_outside:
            needs_investigation = True
            warning = (
                f"УВАГА: Діагнози {persistent_outside} не належать до основного "
                f"кластера симптомів, але залишаються в топі. "
                f"Рекомендовано додаткові дослідження для їх виключення/підтвердження."
            )
        
        return DiagnosisCycleResult(
            top_hypotheses=sorted_hyp,
            top_in_cluster=top_in_cluster,
            top_outside_cluster=top_outside_cluster,
            stop_reason=stop_reason,
            stop_message=stop_message,
            iterations=self.iteration_count,
            questions_asked=self.questions_count,
            duration_seconds=duration,
            present_symptoms=list(self.session_state.known_symptoms),
            absent_symptoms=list(self.session_state.negated_symptoms),
            unknown_symptoms=list(self.session_state.unknown_symptoms),
            needs_additional_investigation=needs_investigation,
            outside_cluster_warning=warning,
            iteration_history=self.iteration_history
        )
    
    def run_full_diagnosis(
        self,
        initial_symptoms: List[str],
        answer_func: Callable[[DiagnosisQuestion], Optional[bool]],
        max_questions: int = 20
    ) -> DiagnosisCycleResult:
        """
        Запустити повний цикл діагностики.
        
        Args:
            initial_symptoms: Початкові симптоми
            answer_func: Функція для отримання відповідей
            max_questions: Максимум питань
        """
        self.start_session(initial_symptoms)
        
        questions_asked = 0
        
        while self.should_continue() and questions_asked < max_questions:
            question = self.get_next_question()
            
            if question is None:
                break
            
            answer = answer_func(question)
            self.process_answer(answer)
            questions_asked += 1
        
        return self.get_result()
    
    @property
    def phase(self) -> SessionPhase:
        return self._phase
    
    @phase.setter
    def phase(self, value: SessionPhase):
        self._phase = value
