"""
Dr.Case — Fuzzy Matcher

Нечітке співставлення тексту з симптомами з бази.

Методи:
- Exact match (точне співпадіння)
- Substring match (підрядок)
- Levenshtein distance (редакційна відстань)
- Token-based similarity (на основі токенів)

Підтримує синоніми українською та англійською.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class MatchResult:
    """Результат співставлення"""
    symptom: str           # Канонічний симптом з бази
    matched_text: str      # Текст, що співпав
    score: float           # Оцінка співпадіння (0-1)
    method: str            # Метод співставлення


class FuzzyMatcher:
    """
    Нечіткий пошук симптомів.
    
    Приклад:
        matcher = FuzzyMatcher(symptom_list)
        results = matcher.match("головний біль")
        # [MatchResult(symptom="Headache", score=0.95, ...)]
    """
    
    # Синоніми: українська → англійська (канонічна форма)
    SYMPTOM_SYNONYMS_UK = {
        # Загальні симптоми
        'головний біль': 'Headache',
        'головная боль': 'Headache',
        'болить голова': 'Headache',
        'біль у голові': 'Headache',
        'мігрень': 'Headache',
        
        'температура': 'Fever',
        'жар': 'Fever',
        'гарячка': 'Fever',
        'лихоманка': 'Fever',
        'підвищена температура': 'Fever',
        'висока температура': 'High Fever',
        
        'кашель': 'Cough',
        'кашляю': 'Cough',
        'сухий кашель': 'Dry Cough',
        'мокрий кашель': 'Productive Cough',
        'вологий кашель': 'Productive Cough',
        
        'нежить': 'Runny Nose',
        'соплі': 'Runny Nose',
        'закладений ніс': 'Nasal Congestion',
        'заложеність носа': 'Nasal Congestion',
        'не дихає ніс': 'Nasal Congestion',
        
        'біль у горлі': 'Sore Throat',
        'болить горло': 'Sore Throat',
        'горло болить': 'Sore Throat',
        'першіння в горлі': 'Sore Throat',
        
        'слабкість': 'Fatigue',
        'втома': 'Fatigue',
        'стомленість': 'Fatigue',
        'немає сил': 'Fatigue',
        'знесилення': 'Fatigue',
        
        'нудота': 'Nausea',
        'нудить': 'Nausea',
        'підкочує': 'Nausea',
        
        'блювота': 'Vomiting',
        'блювання': 'Vomiting',
        'рвота': 'Vomiting',
        
        'діарея': 'Diarrhea',
        'пронос': 'Diarrhea',
        'рідкий стілець': 'Diarrhea',
        
        'запор': 'Constipation',
        'закреп': 'Constipation',
        
        'запаморочення': 'Dizziness',
        'паморочиться': 'Dizziness',
        'крутиться голова': 'Dizziness',
        'голова крутиться': 'Dizziness',
        
        'задишка': 'Shortness of Breath',
        'важко дихати': 'Shortness of Breath',
        'не вистачає повітря': 'Shortness of Breath',
        'задуха': 'Shortness of Breath',
        
        'біль у грудях': 'Chest Pain',
        'болить груди': 'Chest Pain',
        'біль за грудиною': 'Chest Pain',
        
        'біль у животі': 'Abdominal Pain',
        'болить живіт': 'Abdominal Pain',
        'живіт болить': 'Abdominal Pain',
        'біль у шлунку': 'Stomach Pain',
        
        'біль у спині': 'Back Pain',
        'болить спина': 'Back Pain',
        'біль у попереку': 'Lower Back Pain',
        
        'біль у суглобах': 'Joint Pain',
        'суглоби болять': 'Joint Pain',
        'ломить суглоби': 'Joint Pain',
        
        "біль у м'язах": 'Muscle Pain',
        "м'язи болять": 'Muscle Pain',
        'ломота': 'Body Ache',
        'ломить тіло': 'Body Ache',
        
        'висип': 'Rash',
        'висипання': 'Rash',
        'шкіра': 'Skin Rash',
        'свербіж': 'Itching',
        'свербить': 'Itching',
        'чешеться': 'Itching',
        
        'набряк': 'Swelling',
        'набрякло': 'Swelling',
        'опухло': 'Swelling',
        
        'втрата апетиту': 'Loss of Appetite',
        'немає апетиту': 'Loss of Appetite',
        'не хочеться їсти': 'Loss of Appetite',
        
        'втрата ваги': 'Weight Loss',
        'схуднення': 'Weight Loss',
        'схуд': 'Weight Loss',
        
        'безсоння': 'Insomnia',
        'не можу заснути': 'Insomnia',
        'поганий сон': 'Sleep Disturbance',
        
        'тривога': 'Anxiety',
        'тривожність': 'Anxiety',
        'хвилювання': 'Anxiety',
        
        'депресія': 'Depression',
        'пригніченість': 'Depression',
        'поганий настрій': 'Low Mood',
        
        'втрата нюху': 'Loss of Smell',
        'не відчуваю запахів': 'Loss of Smell',
        'аносмія': 'Loss of Smell',
        
        'втрата смаку': 'Loss of Taste',
        'не відчуваю смак': 'Loss of Taste',
        
        'озноб': 'Chills',
        'морозить': 'Chills',
        'тремтіння': 'Chills',
        
        'пітливість': 'Sweating',
        'потію': 'Sweating',
        'нічна пітливість': 'Night Sweats',
        
        'серцебиття': 'Palpitations',
        'тахікардія': 'Rapid Heartbeat',
        'швидке серцебиття': 'Rapid Heartbeat',
        
        'кров у сечі': 'Blood in Urine',
        'кровяниста сеча': 'Blood in Urine',
        
        'часте сечовипускання': 'Frequent Urination',
        'часто ходжу в туалет': 'Frequent Urination',
        
        'біль при сечовипусканні': 'Painful Urination',
        'печіння при сечовипусканні': 'Burning Urination',
        
        'жовтяниця': 'Jaundice',
        'жовта шкіра': 'Jaundice',
        'пожовтіння': 'Jaundice',
        
        'кровотеча': 'Bleeding',
        'кров': 'Bleeding',
        
        'судоми': 'Seizures',
        'конвульсії': 'Convulsions',
        
        'втрата свідомості': 'Loss of Consciousness',
        'непритомність': 'Fainting',
        'знепритомнів': 'Fainting',
        
        'порушення зору': 'Vision Problems',
        'погано бачу': 'Blurred Vision',
        'розмите зображення': 'Blurred Vision',
        
        'біль у вухах': 'Ear Pain',
        'болять вуха': 'Ear Pain',
        'шум у вухах': 'Tinnitus',
        
        'порушення слуху': 'Hearing Loss',
        'погано чую': 'Hearing Loss',
    }
    
    # Англійські синоніми (варіації написання)
    SYMPTOM_SYNONYMS_EN = {
        'head ache': 'Headache',
        'head pain': 'Headache',
        'migraine': 'Headache',
        
        'high temperature': 'Fever',
        'elevated temperature': 'Fever',
        'febrile': 'Fever',
        
        'coughing': 'Cough',
        'dry coughing': 'Dry Cough',
        
        'runny nose': 'Runny Nose',
        'running nose': 'Runny Nose',
        'stuffy nose': 'Nasal Congestion',
        'blocked nose': 'Nasal Congestion',
        
        'throat pain': 'Sore Throat',
        'painful throat': 'Sore Throat',
        
        'tired': 'Fatigue',
        'tiredness': 'Fatigue',
        'exhaustion': 'Fatigue',
        'weakness': 'Weakness',
        
        'feeling sick': 'Nausea',
        'queasy': 'Nausea',
        
        'throwing up': 'Vomiting',
        'being sick': 'Vomiting',
        
        'loose stools': 'Diarrhea',
        'watery stool': 'Diarrhea',
        
        'dizzy': 'Dizziness',
        'lightheaded': 'Dizziness',
        'vertigo': 'Dizziness',
        
        'breathlessness': 'Shortness of Breath',
        'difficulty breathing': 'Shortness of Breath',
        'hard to breathe': 'Shortness of Breath',
        
        'tummy pain': 'Abdominal Pain',
        'stomach ache': 'Stomach Pain',
        'belly pain': 'Abdominal Pain',
        
        'skin rash': 'Rash',
        'skin eruption': 'Rash',
        
        'itchy': 'Itching',
        'itchiness': 'Itching',
        
        'swollen': 'Swelling',
        'puffy': 'Swelling',
        
        'no appetite': 'Loss of Appetite',
        'not hungry': 'Loss of Appetite',
        
        'cant sleep': 'Insomnia',
        'cannot sleep': 'Insomnia',
        'trouble sleeping': 'Sleep Disturbance',
        
        'anxious': 'Anxiety',
        'worried': 'Anxiety',
        'nervous': 'Anxiety',
        
        'sad': 'Depression',
        'depressed': 'Depression',
        'feeling down': 'Low Mood',
        
        'cant smell': 'Loss of Smell',
        'no smell': 'Loss of Smell',
        
        'cant taste': 'Loss of Taste',
        'no taste': 'Loss of Taste',
        
        'shivering': 'Chills',
        'shaking': 'Chills',
        
        'sweaty': 'Sweating',
        'perspiration': 'Sweating',
        
        'heart racing': 'Palpitations',
        'heart pounding': 'Palpitations',
        
        'passed out': 'Fainting',
        'fainted': 'Fainting',
        'blacked out': 'Loss of Consciousness',
        
        'blurry vision': 'Blurred Vision',
        'vision blurry': 'Blurred Vision',
    }
    
    def __init__(
        self, 
        symptom_list: List[str],
        min_score: float = 0.6,
        use_synonyms: bool = True
    ):
        """
        Args:
            symptom_list: Список канонічних симптомів з бази
            min_score: Мінімальна оцінка для співпадіння
            use_synonyms: Використовувати словник синонімів
        """
        self.symptoms = set(symptom_list)
        self.symptoms_lower = {s.lower(): s for s in symptom_list}
        self.min_score = min_score
        self.use_synonyms = use_synonyms
        
        # Об'єднуємо синоніми
        self.all_synonyms = {}
        if use_synonyms:
            self.all_synonyms.update(self.SYMPTOM_SYNONYMS_UK)
            self.all_synonyms.update(self.SYMPTOM_SYNONYMS_EN)
    
    def match(self, text: str) -> List[MatchResult]:
        """
        Знайти всі симптоми в тексті.
        
        Args:
            text: Текст для пошуку
            
        Returns:
            Список знайдених симптомів
        """
        text_lower = text.lower()
        results = []
        found_symptoms = set()
        
        # 1. Exact match з синонімами (найвищий пріоритет)
        for synonym, canonical in self.all_synonyms.items():
            if synonym in text_lower and canonical not in found_symptoms:
                # Перевіряємо чи canonical є в базі
                if canonical in self.symptoms or canonical.lower() in self.symptoms_lower:
                    actual_symptom = self.symptoms_lower.get(canonical.lower(), canonical)
                    results.append(MatchResult(
                        symptom=actual_symptom,
                        matched_text=synonym,
                        score=1.0,
                        method='synonym'
                    ))
                    found_symptoms.add(canonical)
        
        # 2. Exact match з базою
        for symptom in self.symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in text_lower and symptom not in found_symptoms:
                results.append(MatchResult(
                    symptom=symptom,
                    matched_text=symptom,
                    score=1.0,
                    method='exact'
                ))
                found_symptoms.add(symptom)
        
        # 3. Fuzzy match для кожного слова/фрази
        words = text_lower.split()
        
        # Перевіряємо окремі слова
        for word in words:
            if len(word) < 4:
                continue
            
            best_match = self._find_best_fuzzy_match(word, found_symptoms)
            if best_match and best_match.score >= self.min_score:
                results.append(best_match)
                found_symptoms.add(best_match.symptom)
        
        # Перевіряємо біграми (пари слів)
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            best_match = self._find_best_fuzzy_match(phrase, found_symptoms)
            if best_match and best_match.score >= self.min_score:
                results.append(best_match)
                found_symptoms.add(best_match.symptom)
        
        # Сортуємо за score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def match_single(self, text: str) -> Optional[MatchResult]:
        """
        Знайти найкращий симптом для тексту.
        
        Args:
            text: Текст (зазвичай короткий)
            
        Returns:
            Найкращий результат або None
        """
        results = self.match(text)
        return results[0] if results else None
    
    def _find_best_fuzzy_match(
        self, 
        text: str, 
        exclude: Set[str]
    ) -> Optional[MatchResult]:
        """Знайти найкраще нечітке співпадіння"""
        best_score = 0
        best_symptom = None
        
        text_lower = text.lower()
        
        for symptom in self.symptoms:
            if symptom in exclude:
                continue
            
            symptom_lower = symptom.lower()
            
            # Similarity score
            score = self._similarity(text_lower, symptom_lower)
            
            if score > best_score:
                best_score = score
                best_symptom = symptom
        
        if best_symptom and best_score >= self.min_score:
            return MatchResult(
                symptom=best_symptom,
                matched_text=text,
                score=best_score,
                method='fuzzy'
            )
        
        return None
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Обчислити схожість двох рядків"""
        # SequenceMatcher дає хороші результати для коротких рядків
        return SequenceMatcher(None, s1, s2).ratio()
    
    def add_synonym(self, synonym: str, canonical: str) -> None:
        """Додати новий синонім"""
        self.all_synonyms[synonym.lower()] = canonical
    
    def get_all_synonyms_for(self, symptom: str) -> List[str]:
        """Отримати всі синоніми для симптому"""
        synonyms = []
        for syn, canonical in self.all_synonyms.items():
            if canonical.lower() == symptom.lower():
                synonyms.append(syn)
        return synonyms
