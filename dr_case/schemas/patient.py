"""
Dr.Case — Схеми даних пацієнта

Pydantic моделі для:
- Symptom: окремий симптом з severity
- Patient: базові дані пацієнта
- Vitals: вітальні показники
- CaseRecord: повний клінічний випадок
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class Gender(str, Enum):
    """Стать пацієнта"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class SymptomStatus(str, Enum):
    """Статус симптому"""
    PRESENT = "present"       # симптом присутній
    ABSENT = "absent"         # симптом явно відсутній
    UNKNOWN = "unknown"       # невідомо


class Symptom(BaseModel):
    """
    Симптом пацієнта.
    
    Приклад:
        symptom = Symptom(
            name="fever",
            status=SymptomStatus.PRESENT,
            severity=0.8
        )
    """
    name: str = Field(..., description="Назва симптому (lowercase)")
    status: SymptomStatus = Field(default=SymptomStatus.PRESENT)
    severity: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0,
        description="Інтенсивність симптому [0, 1]"
    )
    duration_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Тривалість симптому в днях"
    )
    notes: Optional[str] = Field(default=None, description="Додаткові нотатки")
    
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Нормалізація назви симптому"""
        return v.strip().lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "fever",
                "status": "present",
                "severity": 0.8,
                "duration_days": 3
            }
        }


class Vitals(BaseModel):
    """
    Вітальні показники пацієнта.
    """
    temperature_c: Optional[float] = Field(
        default=None,
        ge=30.0, le=45.0,
        description="Температура тіла (°C)"
    )
    heart_rate: Optional[int] = Field(
        default=None,
        ge=20, le=300,
        description="Пульс (уд/хв)"
    )
    blood_pressure_systolic: Optional[int] = Field(
        default=None,
        ge=50, le=300,
        description="Систолічний тиск (мм рт.ст.)"
    )
    blood_pressure_diastolic: Optional[int] = Field(
        default=None,
        ge=30, le=200,
        description="Діастолічний тиск (мм рт.ст.)"
    )
    respiratory_rate: Optional[int] = Field(
        default=None,
        ge=5, le=60,
        description="Частота дихання (/хв)"
    )
    spo2: Optional[float] = Field(
        default=None,
        ge=50.0, le=100.0,
        description="Сатурація кисню (%)"
    )
    weight_kg: Optional[float] = Field(
        default=None,
        ge=0.5, le=500.0,
        description="Вага (кг)"
    )
    height_cm: Optional[float] = Field(
        default=None,
        ge=20.0, le=300.0,
        description="Зріст (см)"
    )
    
    @property
    def blood_pressure(self) -> Optional[str]:
        """Тиск у форматі 'систолічний/діастолічний'"""
        if self.blood_pressure_systolic and self.blood_pressure_diastolic:
            return f"{self.blood_pressure_systolic}/{self.blood_pressure_diastolic}"
        return None
    
    @property
    def bmi(self) -> Optional[float]:
        """Індекс маси тіла"""
        if self.weight_kg and self.height_cm:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 1)
        return None


class Patient(BaseModel):
    """
    Базові дані пацієнта.
    """
    patient_id: Optional[str] = Field(default=None, description="ID пацієнта")
    age: Optional[int] = Field(default=None, ge=0, le=150, description="Вік")
    gender: Gender = Field(default=Gender.UNKNOWN, description="Стать")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "age": 35,
                "gender": "male"
            }
        }


class CaseRecord(BaseModel):
    """
    Повний клінічний випадок.
    
    Це основна структура для передачі даних між компонентами системи.
    
    Приклад:
        case = CaseRecord(
            patient=Patient(age=35, gender=Gender.MALE),
            chief_complaint="Головний біль та температура",
            symptoms=[
                Symptom(name="headache", severity=0.7),
                Symptom(name="fever", severity=0.8),
            ]
        )
    """
    case_id: Optional[str] = Field(default=None, description="ID випадку")
    patient: Patient = Field(default_factory=Patient)
    
    # Скарги
    chief_complaint: Optional[str] = Field(
        default=None,
        description="Головна скарга (текст від пацієнта)"
    )
    
    # Симптоми
    symptoms: List[Symptom] = Field(
        default_factory=list,
        description="Список симптомів"
    )
    
    # Вітальні показники
    vitals: Optional[Vitals] = Field(default=None)
    
    # Історія хвороби
    medical_history: List[str] = Field(
        default_factory=list,
        description="Попередні захворювання"
    )
    current_medications: List[str] = Field(
        default_factory=list,
        description="Поточні ліки"
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="Алергії"
    )
    
    # Метадані
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default=None)
    
    @property
    def present_symptoms(self) -> List[Symptom]:
        """Симптоми зі статусом PRESENT"""
        return [s for s in self.symptoms if s.status == SymptomStatus.PRESENT]
    
    @property
    def absent_symptoms(self) -> List[Symptom]:
        """Симптоми зі статусом ABSENT"""
        return [s for s in self.symptoms if s.status == SymptomStatus.ABSENT]
    
    @property
    def present_symptom_names(self) -> List[str]:
        """Назви присутніх симптомів"""
        return [s.name for s in self.present_symptoms]
    
    @property
    def absent_symptom_names(self) -> List[str]:
        """Назви відсутніх симптомів"""
        return [s.name for s in self.absent_symptoms]
    
    def add_symptom(
        self, 
        name: str, 
        status: SymptomStatus = SymptomStatus.PRESENT,
        severity: Optional[float] = None
    ) -> None:
        """Додати симптом"""
        # Перевіряємо чи симптом вже є
        for s in self.symptoms:
            if s.name == name.strip().lower():
                s.status = status
                if severity is not None:
                    s.severity = severity
                return
        
        # Додаємо новий
        self.symptoms.append(Symptom(name=name, status=status, severity=severity))
        self.updated_at = datetime.now()
    
    def negate_symptom(self, name: str) -> None:
        """Позначити симптом як відсутній"""
        self.add_symptom(name, status=SymptomStatus.ABSENT)
    
    def get_symptom_severities(self) -> Dict[str, float]:
        """Отримати словник {symptom: severity} для присутніх симптомів"""
        result = {}
        for s in self.present_symptoms:
            result[s.name] = s.severity if s.severity is not None else 1.0
        return result
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "CASE001",
                "patient": {"age": 35, "gender": "male"},
                "chief_complaint": "Головний біль та температура вже 3 дні",
                "symptoms": [
                    {"name": "headache", "status": "present", "severity": 0.7},
                    {"name": "fever", "status": "present", "severity": 0.8},
                    {"name": "rash", "status": "absent"}
                ]
            }
        }
