"""
Dr.Case — Інтелектуальна система медичної діагностики

Архітектура: SOM + Multilabel NN + Циклічний процес діагностики

Модулі:
- config: Конфігурація системи
- encoding: Векторизація симптомів та діагнозів
- som: Self-Organizing Map
- candidate_selector: Відбір кандидатів
- pseudo_generation: Генерація псевдопацієнтів
- multilabel_nn: Нейромережа ранжування
- question_engine: Механізм питань
- diagnosis_cycle: Циклічний процес
- validation: Метрики якості
- api: Backend API
- web_ui: Веб-інтерфейс
"""

__version__ = "0.1.0"
__author__ = "Oleksii Bychkov"

from .config import DrCaseConfig, get_default_config
