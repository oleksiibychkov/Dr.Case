"""
Dr.Case — NLP модуль

Витягування симптомів з природного тексту.

Компоненти:
- TextPreprocessor: Очистка та нормалізація тексту
- FuzzyMatcher: Нечітке співставлення з базою симптомів
- SymptomExtractor: Головний екстрактор симптомів

Приклад використання:
    from dr_case.nlp import SymptomExtractor
    
    extractor = SymptomExtractor.from_database("data/database.json")
    
    result = extractor.extract("Болить голова вже 3 дні, температура 38.5")
    
    print(result.symptoms)      # ['Headache', 'Fever']
    print(result.vitals)        # VitalSigns(temperature=38.5)
    print(result.duration)      # SymptomDuration(days=3)
    print(result.confidence)    # 0.95

Швидкі функції:
    from dr_case.nlp import extract_symptoms, extract_full
    
    symptoms = extract_symptoms("головний біль і нудота", "data/database.json")
    # ['Headache', 'Nausea']
    
    full = extract_full("температура 38, кашель 2 дні", "data/database.json")
    # {'symptoms': [...], 'vitals': {...}, 'duration': {...}}
"""

from .text_preprocessor import (
    TextPreprocessor,
    PreprocessedText,
    Language,
)

from .fuzzy_matcher import (
    FuzzyMatcher,
    MatchResult,
)

from .symptom_extractor import (
    SymptomExtractor,
    ExtractionResult,
    VitalSigns,
    SymptomDuration,
    extract_symptoms,
    extract_full,
)


__all__ = [
    # Preprocessor
    'TextPreprocessor',
    'PreprocessedText',
    'Language',
    
    # Matcher
    'FuzzyMatcher',
    'MatchResult',
    
    # Extractor
    'SymptomExtractor',
    'ExtractionResult',
    'VitalSigns',
    'SymptomDuration',
    
    # Quick functions
    'extract_symptoms',
    'extract_full',
]
