"""
Dr.Case — Symptom Extractor

Головний модуль для витягування симптомів з природного тексту.

Функціональність:
- Витягування симптомів з опису скарг пацієнта
- Підтримка української та англійської мов
- Нечітке співставлення з базою симптомів
- Витягування числових показників (температура, тиск, пульс)
- Витягування тривалості симптомів

Приклад:
    extractor = SymptomExtractor.from_database("data/database.json")
    result = extractor.extract("Болить голова вже 3 дні, температура 38.5")
    
    print(result.symptoms)  # ['Headache', 'Fever']
    print(result.vitals)    # {'temperature': 38.5}
    print(result.duration)  # {'days': 3}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

from .text_preprocessor import TextPreprocessor, PreprocessedText, Language
from .fuzzy_matcher import FuzzyMatcher, MatchResult


@dataclass
class VitalSigns:
    """Вітальні показники"""
    temperature: Optional[float] = None
    pulse: Optional[int] = None
    systolic: Optional[int] = None
    diastolic: Optional[int] = None
    spo2: Optional[int] = None
    respiratory_rate: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def is_empty(self) -> bool:
        return all(v is None for v in self.__dict__.values())


@dataclass
class SymptomDuration:
    """Тривалість симптомів"""
    days: Optional[int] = None
    weeks: Optional[int] = None
    months: Optional[int] = None
    hours: Optional[int] = None
    
    def to_dict(self) -> Dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def to_days(self) -> Optional[float]:
        """Конвертувати в дні"""
        total = 0
        if self.hours:
            total += self.hours / 24
        if self.days:
            total += self.days
        if self.weeks:
            total += self.weeks * 7
        if self.months:
            total += self.months * 30
        return total if total > 0 else None


@dataclass
class ExtractionResult:
    """Результат витягування симптомів"""
    # Вхідний текст
    original_text: str
    
    # Знайдені симптоми
    symptoms: List[str] = field(default_factory=list)
    
    # Деталі співставлення
    matches: List[MatchResult] = field(default_factory=list)
    
    # Заперечені симптоми ("не болить голова")
    negated_symptoms: List[str] = field(default_factory=list)
    
    # Вітальні показники
    vitals: VitalSigns = field(default_factory=VitalSigns)
    
    # Тривалість
    duration: SymptomDuration = field(default_factory=SymptomDuration)
    
    # Мова тексту
    language: Language = Language.ENGLISH
    
    # Впевненість (середня оцінка співставлень)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник"""
        return {
            'original_text': self.original_text,
            'symptoms': self.symptoms,
            'negated_symptoms': self.negated_symptoms,
            'vitals': self.vitals.to_dict(),
            'duration': self.duration.to_dict(),
            'language': self.language.value,
            'confidence': self.confidence,
            'matches': [
                {
                    'symptom': m.symptom,
                    'matched_text': m.matched_text,
                    'score': m.score,
                    'method': m.method
                }
                for m in self.matches
            ]
        }


class SymptomExtractor:
    """
    Витягування симптомів з тексту.
    
    Приклад:
        # Ініціалізація з бази
        extractor = SymptomExtractor.from_database("data/database.json")
        
        # Або з списку симптомів
        extractor = SymptomExtractor(symptom_list=["Headache", "Fever", ...])
        
        # Витягування
        result = extractor.extract("У мене болить голова і температура 38")
        
        print(result.symptoms)     # ['Headache', 'Fever']
        print(result.vitals)       # VitalSigns(temperature=38.0)
        print(result.confidence)   # 0.95
    """
    
    # Патерни заперечення
    NEGATION_PATTERNS_UK = [
        r'не\s+', r'без\s+', r'немає\s+', r'відсутн\w+\s+',
        r'жодн\w+\s+', r'ніяк\w+\s+',
    ]
    
    NEGATION_PATTERNS_EN = [
        r'no\s+', r'not\s+', r'without\s+', r'absence\s+of\s+',
        r"don't\s+have\s+", r"doesn't\s+have\s+",
        r'never\s+', r'neither\s+',
    ]
    
    def __init__(
        self,
        symptom_list: List[str],
        min_confidence: float = 0.6,
        use_synonyms: bool = True
    ):
        """
        Args:
            symptom_list: Список канонічних симптомів
            min_confidence: Мінімальна впевненість для симптому
            use_synonyms: Використовувати синоніми
        """
        self.symptom_list = symptom_list
        self.min_confidence = min_confidence
        
        self.preprocessor = TextPreprocessor(remove_stopwords=False)
        self.matcher = FuzzyMatcher(
            symptom_list=symptom_list,
            min_score=min_confidence,
            use_synonyms=use_synonyms
        )
    
    @classmethod
    def from_database(
        cls, 
        database_path: str,
        min_confidence: float = 0.6,
        use_synonyms: bool = True
    ) -> 'SymptomExtractor':
        """
        Створити екстрактор з бази даних.
        
        Args:
            database_path: Шлях до JSON бази хвороб
            min_confidence: Мінімальна впевненість
            use_synonyms: Використовувати синоніми
        """
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        # Збираємо всі унікальні симптоми
        all_symptoms = set()
        for disease_data in database.values():
            symptoms = disease_data.get('symptoms', [])
            all_symptoms.update(symptoms)
        
        symptom_list = sorted(list(all_symptoms))
        
        return cls(
            symptom_list=symptom_list,
            min_confidence=min_confidence,
            use_synonyms=use_synonyms
        )
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Витягнути симптоми з тексту.
        
        Args:
            text: Текст з описом скарг
            
        Returns:
            ExtractionResult з симптомами та метаданими
        """
        if not text or not text.strip():
            return ExtractionResult(original_text=text)
        
        # Препроцесинг
        preprocessed = self.preprocessor.process(text)
        
        # Витягуємо вітальні показники
        vitals = self._extract_vitals(text)
        
        # Витягуємо тривалість
        duration = self._extract_duration(text)
        
        # Знаходимо симптоми
        matches = self.matcher.match(preprocessed.normalized)
        
        # Перевіряємо на заперечення
        symptoms = []
        negated = []
        
        for match in matches:
            if self._is_negated(match.matched_text, preprocessed.normalized, preprocessed.language):
                negated.append(match.symptom)
            else:
                symptoms.append(match.symptom)
        
        # Обчислюємо впевненість
        confidence = 0.0
        if matches:
            confidence = sum(m.score for m in matches) / len(matches)
        
        return ExtractionResult(
            original_text=text,
            symptoms=symptoms,
            matches=matches,
            negated_symptoms=negated,
            vitals=vitals,
            duration=duration,
            language=preprocessed.language,
            confidence=confidence
        )
    
    def extract_symptoms_only(self, text: str) -> List[str]:
        """
        Швидке витягування тільки списку симптомів.
        
        Args:
            text: Текст
            
        Returns:
            Список симптомів
        """
        result = self.extract(text)
        return result.symptoms
    
    def _extract_vitals(self, text: str) -> VitalSigns:
        """Витягування вітальних показників"""
        vitals = VitalSigns()
        
        numbers = self.preprocessor.extract_numbers(text)
        
        for value, unit in numbers:
            if unit == 'temperature':
                # Перевіряємо розумність температури
                if 35 <= value <= 43:
                    vitals.temperature = value
            elif unit == 'pulse':
                if 30 <= value <= 250:
                    vitals.pulse = int(value)
            elif unit == 'systolic':
                if 60 <= value <= 260:
                    vitals.systolic = int(value)
            elif unit == 'diastolic':
                if 30 <= value <= 160:
                    vitals.diastolic = int(value)
        
        return vitals
    
    def _extract_duration(self, text: str) -> SymptomDuration:
        """Витягування тривалості"""
        duration = SymptomDuration()
        
        durations = self.preprocessor.extract_duration(text)
        
        for value, unit in durations:
            if unit == 'days':
                duration.days = (duration.days or 0) + value
            elif unit == 'weeks':
                duration.weeks = (duration.weeks or 0) + value
            elif unit == 'months':
                duration.months = (duration.months or 0) + value
            elif unit == 'hours':
                duration.hours = (duration.hours or 0) + value
        
        return duration
    
    def _is_negated(
        self, 
        matched_text: str, 
        full_text: str, 
        language: Language
    ) -> bool:
        """Перевірити чи симптом заперечений"""
        import re
        
        patterns = (
            self.NEGATION_PATTERNS_UK if language == Language.UKRAINIAN 
            else self.NEGATION_PATTERNS_EN
        )
        
        # Шукаємо заперечення перед симптомом
        for pattern in patterns:
            neg_pattern = f'{pattern}\\w*\\s*{re.escape(matched_text)}'
            if re.search(neg_pattern, full_text, re.IGNORECASE):
                return True
        
        return False
    
    def add_synonym(self, synonym: str, canonical_symptom: str) -> None:
        """
        Додати новий синонім.
        
        Args:
            synonym: Синонім (напр. "болить голова")
            canonical_symptom: Канонічний симптом (напр. "Headache")
        """
        self.matcher.add_synonym(synonym, canonical_symptom)
    
    def get_supported_symptoms(self) -> List[str]:
        """Отримати список підтримуваних симптомів"""
        return sorted(self.symptom_list)
    
    def get_symptom_count(self) -> int:
        """Кількість симптомів у базі"""
        return len(self.symptom_list)


# Зручні функції для швидкого використання

def extract_symptoms(
    text: str, 
    database_path: str,
    min_confidence: float = 0.6
) -> List[str]:
    """
    Швидка функція для витягування симптомів.
    
    Args:
        text: Текст з описом
        database_path: Шлях до бази
        min_confidence: Мінімальна впевненість
        
    Returns:
        Список симптомів
    """
    extractor = SymptomExtractor.from_database(database_path, min_confidence)
    return extractor.extract_symptoms_only(text)


def extract_full(
    text: str, 
    database_path: str,
    min_confidence: float = 0.6
) -> Dict[str, Any]:
    """
    Повне витягування з усіма метаданими.
    
    Returns:
        Словник з симптомами, вітальними показниками, тривалістю
    """
    extractor = SymptomExtractor.from_database(database_path, min_confidence)
    result = extractor.extract(text)
    return result.to_dict()
