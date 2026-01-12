"""
Dr.Case — Text Preprocessor

Очистка та нормалізація тексту перед витягуванням симптомів.

Підтримує:
- Українську мову
- Англійську мову
- Очистка від зайвих символів
- Нормалізація пробілів
- Токенізація
"""

import re
import unicodedata
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Підтримувані мови"""
    ENGLISH = "en"
    UKRAINIAN = "uk"
    AUTO = "auto"


@dataclass
class PreprocessedText:
    """Результат препроцесингу"""
    original: str
    normalized: str
    tokens: List[str]
    language: Language
    sentences: List[str]


class TextPreprocessor:
    """
    Препроцесор тексту для витягування симптомів.
    
    Приклад:
        preprocessor = TextPreprocessor()
        result = preprocessor.process("У мене болить голова і температура 38")
        print(result.tokens)  # ['мене', 'болить', 'голова', 'температура', '38']
    """
    
    # Стоп-слова українською
    STOP_WORDS_UK = {
        'і', 'та', 'а', 'але', 'або', 'що', 'як', 'це', 'той', 'та',
        'в', 'у', 'на', 'з', 'із', 'до', 'від', 'по', 'за', 'над', 'під',
        'я', 'ми', 'ви', 'він', 'вона', 'воно', 'вони', 'мене', 'мені',
        'є', 'був', 'була', 'було', 'були', 'буде', 'будуть',
        'дуже', 'трохи', 'мало', 'багато', 'ще', 'вже', 'теж', 'також',
        'коли', 'якщо', 'тому', 'бо', 'тоді', 'потім', 'зараз',
        'так', 'ні', 'не', 'ось', 'там', 'тут', 'де', 'куди',
        'можна', 'треба', 'потрібно', 'хочу', 'маю',
    }
    
    # Стоп-слова англійською
    STOP_WORDS_EN = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
        'feel', 'feeling', 'felt', 'get', 'getting', 'got',
        'really', 'also', 'still', 'since', 'been', 'having',
    }
    
    # Кирилиця для детекції мови
    CYRILLIC_PATTERN = re.compile(r'[\u0400-\u04FF]')
    
    def __init__(self, remove_stopwords: bool = True):
        """
        Args:
            remove_stopwords: Чи видаляти стоп-слова
        """
        self.remove_stopwords = remove_stopwords
    
    def process(
        self, 
        text: str, 
        language: Language = Language.AUTO
    ) -> PreprocessedText:
        """
        Повний препроцесинг тексту.
        
        Args:
            text: Вхідний текст
            language: Мова (auto для автовизначення)
            
        Returns:
            PreprocessedText з результатами
        """
        if not text or not text.strip():
            return PreprocessedText(
                original=text,
                normalized="",
                tokens=[],
                language=Language.ENGLISH,
                sentences=[]
            )
        
        # Визначаємо мову
        if language == Language.AUTO:
            language = self._detect_language(text)
        
        # Нормалізація
        normalized = self._normalize(text)
        
        # Розбиття на речення
        sentences = self._split_sentences(normalized)
        
        # Токенізація
        tokens = self._tokenize(normalized, language)
        
        return PreprocessedText(
            original=text,
            normalized=normalized,
            tokens=tokens,
            language=language,
            sentences=sentences
        )
    
    def _detect_language(self, text: str) -> Language:
        """Автовизначення мови"""
        if self.CYRILLIC_PATTERN.search(text):
            return Language.UKRAINIAN
        return Language.ENGLISH
    
    def _normalize(self, text: str) -> str:
        """Нормалізація тексту"""
        # Unicode нормалізація
        text = unicodedata.normalize('NFKC', text)
        
        # Lowercase
        text = text.lower()
        
        # Заміна різних типів пробілів на звичайний
        text = re.sub(r'[\t\n\r\f\v]+', ' ', text)
        
        # Видалення зайвих пробілів
        text = re.sub(r'\s+', ' ', text)
        
        # Видалення спецсимволів (залишаємо букви, цифри, базову пунктуацію)
        text = re.sub(r'[^\w\s\.,!?;:\-\'\"°℃]', ' ', text)
        
        # Нормалізація пунктуації
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Розбиття на речення"""
        # Простий розбив по .!?
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str, language: Language) -> List[str]:
        """Токенізація"""
        # Видаляємо пунктуацію для токенізації
        text_clean = re.sub(r'[.,!?;:\-\'\"]+', ' ', text)
        
        # Розбиваємо на слова
        tokens = text_clean.split()
        
        # Фільтруємо
        tokens = [t for t in tokens if len(t) > 1]
        
        # Видаляємо стоп-слова
        if self.remove_stopwords:
            stop_words = (
                self.STOP_WORDS_UK if language == Language.UKRAINIAN 
                else self.STOP_WORDS_EN
            )
            tokens = [t for t in tokens if t not in stop_words]
        
        return tokens
    
    def extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """
        Витягування числових значень з одиницями.
        
        Returns:
            Список (значення, одиниця/контекст)
        """
        results = []
        
        # Температура
        temp_patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:°[CcСс]|градус|degrees?|celsius)',
            r'температур[аи]?\s*[:\-]?\s*(\d+(?:[.,]\d+)?)',
            r'temperature\s*[:\-]?\s*(\d+(?:[.,]\d+)?)',
        ]
        
        for pattern in temp_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = float(match.group(1).replace(',', '.'))
                results.append((value, 'temperature'))
        
        # Тиск
        pressure_pattern = r'(\d+)\s*/\s*(\d+)'
        for match in re.finditer(pressure_pattern, text):
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            results.append((systolic, 'systolic'))
            results.append((diastolic, 'diastolic'))
        
        # Пульс
        pulse_patterns = [
            r'пульс\s*[:\-]?\s*(\d+)',
            r'pulse\s*[:\-]?\s*(\d+)',
            r'(\d+)\s*(?:уд/?хв|bpm)',
        ]
        
        for pattern in pulse_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = int(match.group(1))
                results.append((value, 'pulse'))
        
        return results
    
    def extract_duration(self, text: str) -> List[Tuple[int, str]]:
        """
        Витягування тривалості симптомів.
        
        Returns:
            Список (кількість, одиниця часу)
        """
        results = []
        
        # Українські патерни
        uk_patterns = [
            (r'(\d+)\s*(?:дн[іяь]в?|день)', 'days'),
            (r'(\d+)\s*(?:тижн[іяь]в?|тиждень)', 'weeks'),
            (r'(\d+)\s*(?:місяц[іяь]в?|місяць)', 'months'),
            (r'(\d+)\s*(?:годин[иу]?|год)', 'hours'),
            (r'(\d+)\s*(?:хвилин[иу]?|хв)', 'minutes'),
        ]
        
        # Англійські патерни
        en_patterns = [
            (r'(\d+)\s*days?', 'days'),
            (r'(\d+)\s*weeks?', 'weeks'),
            (r'(\d+)\s*months?', 'months'),
            (r'(\d+)\s*hours?', 'hours'),
            (r'(\d+)\s*minutes?', 'minutes'),
        ]
        
        for pattern, unit in uk_patterns + en_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = int(match.group(1))
                results.append((value, unit))
        
        return results
