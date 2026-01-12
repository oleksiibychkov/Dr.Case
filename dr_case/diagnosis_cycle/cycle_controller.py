"""
Dr.Case — Головний контролер циклічного процесу діагностики

Інтегрує всі компоненти:
- SOM + CandidateSelector
- TwoBranchNN
- Question Engine (InformationGainCalculator, QuestionSelector, AnswerProcessor)
- Stopping Criteria
- Hypothesis Tracker

Реалізує алгоритм діагностики:
1. Пацієнт → Симптоми → SOM → Кандидати → NN → Гіпотези
2. Перевірка критеріїв зупинки
3. Якщо CONTINUE → вибір питання за EIG → відповідь → оновлення
4. Repeat until STOP
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from .stopping_criteria import StoppingCriteria, StoppingConfig, StopReason, StopDecision
from .hypothesis_tracker import HypothesisTracker


class SessionPhase(Enum):
    """Фаза сесії"""
    INITIAL = "initial"           # Початкові симптоми
    QUESTIONING = "questioning"   # Уточнюючі питання
    NEED_TESTS = "need_tests"     # Потрібні лаб. тести
    AWAITING_RESULTS = "awaiting_results"  # Очікування результатів
    COMPLETED = "completed"       # Завершено
    FAILED = "failed"             # Помилка


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
class IterationResult:
    """Результат однієї ітерації"""
    iteration: int
    
    # Вектор симптомів
    symptom_vector: np.ndarray
    
    # SOM результат
    active_units: List[int]
    membership: Dict[int, float]
    
    # Кандидати
    candidates: List[str]
    
    # Гіпотези від NN
    hypotheses: Dict[str, float]
    
    # Питання (якщо є)
    question: Optional[DiagnosisQuestion] = None
    
    # Критерій зупинки
    stop_decision: Optional[StopDecision] = None


@dataclass
class DiagnosisCycleResult:
    """Фінальний результат діагностики"""
    # Топ гіпотези
    top_hypotheses: List[Tuple[str, float]]
    
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
    
    # Рекомендації (якщо NEED_TEST)
    recommended_tests: List[str] = field(default_factory=list)
    
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
    
    Приклад використання:
        # Ініціалізація
        controller = DiagnosisCycleController.from_models(
            database_path="data/unified_disease_symptom_merged.json",
            som_path="models/som_merged.pkl",
            nn_path="models/nn_two_branch.pt"
        )
        
        # Автоматична діагностика (з функцією відповідей)
        def answer_func(question: DiagnosisQuestion) -> Optional[bool]:
            print(f"Q: {question.text}")
            ans = input("(y/n/skip): ")
            return {'y': True, 'n': False}.get(ans, None)
        
        result = controller.run_full_diagnosis(
            initial_symptoms=['Fever', 'Cough', 'Headache'],
            answer_func=answer_func
        )
        
        print(f"Діагноз: {result.top_diagnosis} ({result.top_confidence:.1%})")
        
        # АБО: Покроковий режим
        controller.start_session(['Fever', 'Cough'])
        
        while controller.should_continue():
            question = controller.get_next_question()
            # ... отримати відповідь від користувача
            controller.process_answer(answer=True)
        
        result = controller.get_result()
    """
    
    def __init__(
        self,
        database: Dict[str, Any],
        som_model: Any,
        nn_model: Any,
        symptom_vocab: Any,
        stopping_config: Optional[StoppingConfig] = None,
        language: str = "uk"
    ):
        """
        Args:
            database: База хвороб-симптомів
            som_model: Навчена SOM модель
            nn_model: Навчена TwoBranchNN модель
            symptom_vocab: Словник симптомів
            stopping_config: Конфігурація критеріїв зупинки
            language: Мова питань (uk/en)
        """
        self.database = database
        self.som_model = som_model
        self.nn_model = nn_model
        self.symptom_vocab = symptom_vocab
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
        
        # Зберігаємо AnswerType для використання
        self._AnswerType = AnswerType
        
        self.ig_calculator = InformationGainCalculator.from_database(self.database)
        
        # QuestionGenerator з мовою
        question_generator = QuestionGenerator(language=self.language)
        
        self.question_selector = QuestionSelector(
            self.ig_calculator,
            question_generator=question_generator
        )
        self.answer_processor = AnswerProcessor(self.ig_calculator)
    
    def _reset_session(self):
        """Скинути поточну сесію"""
        from dr_case.question_engine import SessionState
        
        self.session_state = SessionState()
        self.hypothesis_tracker = HypothesisTracker()
        self.iteration_history: List[IterationResult] = []
        self.current_iteration = 0
        self.phase = SessionPhase.INITIAL
        self.start_time = datetime.now()
        self.current_question: Optional[DiagnosisQuestion] = None
        self._current_candidates: List[str] = []
        self._current_hypotheses: Dict[str, float] = {}
    
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
            database_path: Шлях до бази даних JSON
            som_path: Шлях до SOM моделі (.pkl)
            nn_path: Шлях до NN моделі (.pt)
            stopping_config: Конфігурація зупинки
            language: Мова
            
        Returns:
            DiagnosisCycleController
        """
        import torch
        
        # Завантажуємо базу
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        # Завантажуємо SOM
        with open(som_path, 'rb') as f:
            som_data = pickle.load(f)
        
        # Debug: показуємо структуру SOM checkpoint
        if isinstance(som_data, dict):
            print(f"   SOM checkpoint keys: {list(som_data.keys())}")
            
            # Структура з SOMModel.save():
            # - som: MiniSom об'єкт
            # - unit_to_diseases: {unit_idx: [diseases]}
            # - disease_names, config, qe, te, etc.
            som_model = som_data.get('som') or som_data.get('model') or som_data.get('som_model')
            
            # Зберігаємо unit_to_diseases як атрибут моделі
            unit_to_diseases = som_data.get('unit_to_diseases', {})
            if som_model is not None:
                som_model._unit_to_diseases = unit_to_diseases
                print(f"   unit_to_diseases: {len(unit_to_diseases)} units")
        else:
            # som_data може бути безпосередньо MiniSom об'єктом
            print(f"   SOM checkpoint type: {type(som_data)}")
            som_model = som_data
            # Спробуємо перевірити чи вже є атрибут
            if not hasattr(som_model, '_unit_to_diseases'):
                som_model._unit_to_diseases = {}
        
        # Завантажуємо NN
        nn_checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
        
        # Створюємо модель NN
        from dr_case.neural_network.two_branch_model import TwoBranchNN
        
        # Структура checkpoint з retrain_nn.py:
        # - model_state (not model_state_dict)
        # - model_config
        # - disease_names
        # - symptom_names
        
        model_config = nn_checkpoint.get('model_config', {})
        
        # Debug: показуємо що є в checkpoint
        print(f"   NN checkpoint keys: {list(nn_checkpoint.keys())}")
        print(f"   model_config: {model_config}")
        
        # Визначаємо розмірності
        if model_config:
            # model_config може мати різні імена полів
            n_symptoms = (model_config.get('n_symptoms') or 
                         model_config.get('symptom_dim') or 460)
            som_dim = model_config.get('som_dim', 10)
            n_diseases = (model_config.get('n_diseases') or 
                         model_config.get('output_dim') or len(database))
        else:
            # Fallback: спочатку з бази
            all_symptoms = set()
            for disease_data in database.values():
                all_symptoms.update(disease_data.get('symptoms', []))
            n_symptoms = len(all_symptoms)
            n_diseases = len(database)
            som_dim = 10
            
            # Спробуємо витягти з state_dict для точніших значень
            state_dict = nn_checkpoint.get('model_state') or nn_checkpoint.get('model_state_dict')
            if state_dict:
                # Визначаємо розміри з ваг
                for key, value in state_dict.items():
                    if 'symptom_branch.layers.0.0.weight' in key:
                        n_symptoms = value.shape[1]
                    if 'output_layer.weight' in key:
                        n_diseases = value.shape[0]
            
            print(f"   Fallback dimensions: n_symptoms={n_symptoms}, n_diseases={n_diseases}, som_dim={som_dim}")
        
        nn_model = TwoBranchNN(
            n_symptoms=n_symptoms,
            som_dim=som_dim,
            n_diseases=n_diseases
        )
        
        # Завантажуємо ваги (може бути model_state або model_state_dict)
        state_dict = nn_checkpoint.get('model_state') or nn_checkpoint.get('model_state_dict')
        if state_dict:
            nn_model.load_state_dict(state_dict)
        nn_model.eval()
        
        # Зберігаємо disease_names з checkpoint (важливо для правильного порядку!)
        disease_names = nn_checkpoint.get('disease_names', list(database.keys()))
        nn_model.disease_names = disease_names  # Зберігаємо як атрибут моделі
        
        # Словник симптомів (передаємо шлях, не словник)
        from dr_case.encoding import SymptomVocabulary
        symptom_vocab = SymptomVocabulary.from_database(database_path)
        
        return cls(
            database=database,
            som_model=som_model,
            nn_model=nn_model,
            symptom_vocab=symptom_vocab,
            stopping_config=stopping_config,
            language=language
        )
    
    def start_session(self, initial_symptoms: List[str]) -> IterationResult:
        """
        Почати нову сесію діагностики.
        
        Args:
            initial_symptoms: Початкові симптоми
            
        Returns:
            Результат першої ітерації
        """
        self._reset_session()
        
        # Додаємо початкові симптоми
        for symptom in initial_symptoms:
            self.session_state.add_symptom_answer(symptom, self._AnswerType.YES)
        
        self.phase = SessionPhase.QUESTIONING
        
        # Перша ітерація
        return self._run_iteration()
    
    def _run_iteration(self) -> IterationResult:
        """Виконати одну ітерацію діагностики"""
        self.current_iteration += 1
        
        # 1. Отримуємо вектор симптомів
        symptom_vector = self._get_symptom_vector()
        
        # 2. Проєкція на SOM → membership
        active_units, membership = self._project_to_som(symptom_vector)
        
        # 3. Відбір кандидатів
        candidates = self._select_candidates(active_units, membership)
        self._current_candidates = candidates
        
        # 4. NN → гіпотези
        hypotheses = self._predict_hypotheses(symptom_vector, membership)
        
        # BYPASS: Не фільтруємо по кандидатах, бо SOM має високий QE (2.25)
        # NN показує 93% Recall@5 без фільтрації
        # TODO: Перенавчити SOM для кращої кластеризації, тоді повернути фільтрацію
        filtered_hyp = hypotheses  # Використовуємо всі гіпотези від NN
        
        # Нормалізуємо (softmax вже нормалізований)
        total = sum(filtered_hyp.values())
        if total > 0:
            filtered_hyp = {d: p / total for d, p in filtered_hyp.items()}
        
        self._current_hypotheses = filtered_hyp
        
        # 5. Оновлюємо трекер гіпотез
        self.hypothesis_tracker.update(filtered_hyp, self.current_iteration)
        
        # 6. Перевіряємо критерії зупинки
        stop_decision = self.stopping_criteria.check(
            current_hypotheses=filtered_hyp,
            iteration=self.current_iteration,
            questions_asked=len(self.session_state.symptom_history),
            hypothesis_history=self.hypothesis_tracker.get_hypothesis_history()[:-1],
            elapsed_minutes=self._elapsed_minutes(),
            available_questions=self._count_available_questions()
        )
        
        # 7. Вибираємо наступне питання (якщо продовжуємо)
        question = None
        if stop_decision.should_continue:
            question = self._select_next_question(filtered_hyp)
            self.current_question = question
        else:
            self.phase = SessionPhase.COMPLETED
            if stop_decision.reason == StopReason.NEED_TEST:
                self.phase = SessionPhase.NEED_TESTS
        
        # Створюємо результат ітерації
        result = IterationResult(
            iteration=self.current_iteration,
            symptom_vector=symptom_vector,
            active_units=active_units,
            membership=membership,
            candidates=candidates,
            hypotheses=filtered_hyp,
            question=question,
            stop_decision=stop_decision
        )
        
        self.iteration_history.append(result)
        
        return result
    
    def _get_symptom_vector(self) -> np.ndarray:
        """Отримати бінарний вектор симптомів"""
        vector = np.zeros(self.symptom_vocab.size, dtype=np.float32)
        
        for symptom in self.session_state.known_symptoms:
            idx = self.symptom_vocab.symptom_to_index(symptom)
            if idx is not None:
                vector[idx] = 1.0
        
        return vector
    
    def _project_to_som(
        self, 
        symptom_vector: np.ndarray
    ) -> Tuple[List[int], Dict[int, float]]:
        """Проєкція на SOM"""
        # Знаходимо BMU та активні юніти
        from minisom import MiniSom
        
        if isinstance(self.som_model, MiniSom):
            # Пряме використання MiniSom
            bmu = self.som_model.winner(symptom_vector)
            bmu_idx = bmu[0] * self.som_model._weights.shape[1] + bmu[1]
            
            # Обчислюємо відстані до всіх юнітів
            distances = []
            for i in range(self.som_model._weights.shape[0]):
                for j in range(self.som_model._weights.shape[1]):
                    w = self.som_model._weights[i, j]
                    dist = np.linalg.norm(symptom_vector - w)
                    unit_idx = i * self.som_model._weights.shape[1] + j
                    distances.append((unit_idx, dist))
            
            # Сортуємо за відстанню
            distances.sort(key=lambda x: x[1])
            
            # Топ-10 юнітів
            top_k = 10
            active_units = [d[0] for d in distances[:top_k]]
            
            # Membership через softmax
            dists = np.array([d[1] for d in distances[:top_k]])
            lambda_ = 1.0
            exp_dists = np.exp(-dists**2 / lambda_)
            memberships = exp_dists / exp_dists.sum()
            
            membership = {
                active_units[i]: float(memberships[i])
                for i in range(len(active_units))
            }
            
            return active_units, membership
        else:
            # Використовуємо збережену структуру
            raise NotImplementedError("Тільки MiniSom підтримується")
    
    def _select_candidates(
        self,
        active_units: List[int],
        membership: Dict[int, float]
    ) -> List[str]:
        """Відбір кандидатів з активних юнітів"""
        # Отримуємо unit_to_diseases з SOM моделі
        candidates = set()
        
        # Якщо є збережений індекс
        if hasattr(self.som_model, '_unit_to_diseases'):
            unit_to_diseases = self.som_model._unit_to_diseases
        elif hasattr(self.som_model, 'unit_to_diseases'):
            unit_to_diseases = self.som_model.unit_to_diseases
        else:
            # Fallback: повертаємо всі хвороби
            return list(self.database.keys())
        
        # Визначаємо розмір сітки
        h, w = self.som_model._weights.shape[:2]
        
        for unit_idx in active_units:
            # Конвертуємо flat index в tuple (i, j)
            i = unit_idx // w
            j = unit_idx % w
            unit_tuple = (i, j)
            
            # Пробуємо різні формати ключів
            found = False
            
            # Варіант 1: tuple з int
            if unit_tuple in unit_to_diseases:
                candidates.update(unit_to_diseases[unit_tuple])
                found = True
            
            # Варіант 2: flat index
            elif unit_idx in unit_to_diseases:
                candidates.update(unit_to_diseases[unit_idx])
                found = True
            
            # Варіант 3: string
            elif str(unit_tuple) in unit_to_diseases:
                candidates.update(unit_to_diseases[str(unit_tuple)])
                found = True
            
            # Варіант 4: шукаємо tuple з np.int64
            if not found:
                for key in unit_to_diseases.keys():
                    if isinstance(key, tuple) and len(key) == 2:
                        if int(key[0]) == i and int(key[1]) == j:
                            candidates.update(unit_to_diseases[key])
                            break
        
        return list(candidates)
    
    def _predict_hypotheses(
        self,
        symptom_vector: np.ndarray,
        membership: Dict[int, float]
    ) -> Dict[str, float]:
        """Передбачення гіпотез через NN"""
        import torch
        
        # Готуємо вхід для NN
        symptoms_tensor = torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0)
        
        # SOM context (топ-k memberships)
        som_dim = 10  # Має відповідати конфігурації NN
        som_context = np.zeros(som_dim, dtype=np.float32)
        
        sorted_membership = sorted(membership.items(), key=lambda x: x[1], reverse=True)
        for i, (unit_idx, mem) in enumerate(sorted_membership[:som_dim]):
            som_context[i] = mem
        
        som_tensor = torch.tensor(som_context, dtype=torch.float32).unsqueeze(0)
        
        # Передбачення
        with torch.no_grad():
            outputs = self.nn_model(symptoms_tensor, som_tensor)
            # SOFTMAX (не sigmoid!) - бо CrossEntropyLoss при навчанні
            probs = torch.softmax(outputs, dim=-1).squeeze().numpy()
        
        # Маємо мати список disease_names з checkpoint або database
        # Якщо disease_names збережено в моделі - використовуємо їх для правильного порядку
        if hasattr(self.nn_model, 'disease_names') and self.nn_model.disease_names:
            disease_names = self.nn_model.disease_names
        else:
            disease_names = list(self.database.keys())
        
        if len(probs) != len(disease_names):
            # Fallback
            return {d: 1.0 / len(disease_names) for d in disease_names}
        
        return {
            disease_names[i]: float(probs[i])
            for i in range(len(disease_names))
        }
    
    def _select_next_question(
        self,
        hypotheses: Dict[str, float]
    ) -> Optional[DiagnosisQuestion]:
        """Вибрати наступне питання за EIG"""
        # Вибираємо питання з правильними параметрами
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
            explanation=f"EIG: {question.eig:.4f}",  # Question не має explanation
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
        available = len(all_symptoms) - len(excluded)
        
        return max(0, available)
    
    def _elapsed_minutes(self) -> float:
        """Час від початку сесії в хвилинах"""
        return (datetime.now() - self.start_time).total_seconds() / 60.0
    
    def process_answer(self, answer: Optional[bool]) -> IterationResult:
        """
        Обробити відповідь на поточне питання.
        
        Args:
            answer: True = так, False = ні, None = не знаю
            
        Returns:
            Результат наступної ітерації
        """
        if self.current_question is None:
            raise ValueError("Немає поточного питання")
        
        symptom = self.current_question.symptom
        
        # Оновлюємо стан сесії
        if answer is True:
            self.session_state.add_symptom_answer(symptom, self._AnswerType.YES)
        elif answer is False:
            self.session_state.add_symptom_answer(symptom, self._AnswerType.NO)
        else:
            self.session_state.add_symptom_answer(symptom, self._AnswerType.UNKNOWN)
        
        # Наступна ітерація
        return self._run_iteration()
    
    def get_next_question(self) -> Optional[DiagnosisQuestion]:
        """Отримати поточне питання"""
        return self.current_question
    
    def should_continue(self) -> bool:
        """Чи потрібно продовжувати"""
        return self.phase == SessionPhase.QUESTIONING and self.current_question is not None
    
    def get_current_hypotheses(self) -> Dict[str, float]:
        """Отримати поточні гіпотези"""
        return self._current_hypotheses.copy()
    
    def get_top_hypotheses(self, n: int = 5) -> List[Tuple[str, float]]:
        """Отримати топ-N гіпотез"""
        sorted_hyp = sorted(
            self._current_hypotheses.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_hyp[:n]
    
    def get_result(self) -> DiagnosisCycleResult:
        """Отримати фінальний результат"""
        if self.iteration_history:
            last = self.iteration_history[-1]
            stop_decision = last.stop_decision
        else:
            stop_decision = StopDecision(
                reason=StopReason.SAFETY_LIMIT,
                should_stop=True,
                message="Немає ітерацій"
            )
        
        return DiagnosisCycleResult(
            top_hypotheses=self.get_top_hypotheses(10),
            stop_reason=stop_decision.reason if stop_decision else StopReason.CONTINUE,
            stop_message=stop_decision.message if stop_decision else "",
            iterations=self.current_iteration,
            questions_asked=len(self.session_state.symptom_history),
            duration_seconds=(datetime.now() - self.start_time).total_seconds(),
            present_symptoms=list(self.session_state.known_symptoms),
            absent_symptoms=list(self.session_state.negated_symptoms),
            unknown_symptoms=list(self.session_state.unknown_symptoms),
            recommended_tests=stop_decision.recommended_tests if stop_decision else [],
            iteration_history=self.iteration_history
        )
    
    def run_full_diagnosis(
        self,
        initial_symptoms: List[str],
        answer_func: Callable[[DiagnosisQuestion], Optional[bool]]
    ) -> DiagnosisCycleResult:
        """
        Запустити повну діагностику.
        
        Args:
            initial_symptoms: Початкові симптоми
            answer_func: Функція для отримання відповідей
            
        Returns:
            DiagnosisCycleResult
        """
        self.start_session(initial_symptoms)
        
        while self.should_continue():
            question = self.get_next_question()
            if question is None:
                break
            
            answer = answer_func(question)
            self.process_answer(answer)
        
        return self.get_result()
    
    def __repr__(self) -> str:
        return (
            f"DiagnosisCycleController("
            f"diseases={len(self.database)}, "
            f"symptoms={self.symptom_vocab.size}, "
            f"phase={self.phase.value})"
        )
