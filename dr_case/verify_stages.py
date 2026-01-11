#!/usr/bin/env python3
"""
Dr.Case — Скрипт перевірки етапів 1-10
Запуск: python verify_stages.py [шлях_до_dr_case]

Якщо шлях не вказано, шукає dr_case в поточній директорії.
"""

import sys
import os
import json
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import traceback

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФІГУРАЦІЯ ПЕРЕВІРКИ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Optional[str] = None

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def ok(msg): return f"{Colors.GREEN}✓{Colors.END} {msg}"
def fail(msg): return f"{Colors.RED}✗{Colors.END} {msg}"
def warn(msg): return f"{Colors.YELLOW}⚠{Colors.END} {msg}"
def info(msg): return f"{Colors.BLUE}ℹ{Colors.END} {msg}"
def header(msg): return f"\n{Colors.BOLD}{Colors.CYAN}{'═'*60}\n{msg}\n{'═'*60}{Colors.END}"

# ═══════════════════════════════════════════════════════════════════════════════
# ДОПОМІЖНІ ФУНКЦІЇ
# ═══════════════════════════════════════════════════════════════════════════════

def check_file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()

def check_dir_exists(path: Path) -> bool:
    return path.exists() and path.is_dir()

def try_import_module(module_path: Path, module_name: str) -> Tuple[bool, Any, str]:
    """Спроба імпортувати модуль. Повертає (success, module, error_msg)"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return False, None, f"Не вдалося завантажити spec для {module_path}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return True, module, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def check_class_exists(module, class_name: str) -> bool:
    return hasattr(module, class_name)

def check_function_exists(module, func_name: str) -> bool:
    return hasattr(module, func_name) and callable(getattr(module, func_name))

def safe_call(func, *args, **kwargs) -> Tuple[bool, Any, str]:
    """Безпечний виклик функції. Повертає (success, result, error_msg)"""
    try:
        result = func(*args, **kwargs)
        return True, result, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# ПЕРЕВІРКА ЕТАПІВ
# ═══════════════════════════════════════════════════════════════════════════════

class StageVerifier:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.results: Dict[str, List[CheckResult]] = {}
        
        # Додаємо project root до sys.path для імпортів
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        if str(self.root.parent) not in sys.path:
            sys.path.insert(0, str(self.root.parent))

    def add_result(self, stage: str, result: CheckResult):
        if stage not in self.results:
            self.results[stage] = []
        self.results[stage].append(result)

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 1: config/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_1_config(self) -> List[CheckResult]:
        """Етап 1: Конфігурація параметрів"""
        stage = "1_config"
        config_dir = self.root / "config"
        
        # 1.1 Перевірка структури
        if not check_dir_exists(config_dir):
            self.add_result(stage, CheckResult("config/", False, "Папка config/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("config/", True, "Папка існує"))
        
        # 1.2 Перевірка файлів
        required_files = ["__init__.py", "default_config.py"]
        optional_files = ["optimized_config.py", "runtime_config.py"]
        
        for f in required_files:
            exists = check_file_exists(config_dir / f)
            self.add_result(stage, CheckResult(f"config/{f}", exists, 
                "Файл існує" if exists else "Файл відсутній (ОБОВ'ЯЗКОВИЙ)"))
        
        for f in optional_files:
            exists = check_file_exists(config_dir / f)
            self.add_result(stage, CheckResult(f"config/{f}", exists, 
                "Файл існує" if exists else "Файл відсутній (опціональний)", 
                details="optional"))
        
        # 1.3 Спроба імпорту
        init_path = config_dir / "__init__.py"
        if check_file_exists(init_path):
            success, module, err = try_import_module(config_dir / "default_config.py", "dr_case.config.default_config")
            if success:
                self.add_result(stage, CheckResult("Імпорт default_config", True, "Успішно"))
                
                # Перевірка наявності конфігураційних словників/класів
                config_names = ["SOM_CONFIG", "CANDIDATE_SELECTOR_CONFIG", "NN_ARCHITECTURE_CONFIG", 
                               "NN_TRAINING_CONFIG", "QUESTION_ENGINE_CONFIG", "STOPPING_CRITERIA_CONFIG",
                               "DrCaseConfig", "load_default_config"]
                
                for name in config_names:
                    exists = hasattr(module, name)
                    self.add_result(stage, CheckResult(f"  {name}", exists,
                        "Визначено" if exists else "Не знайдено"))
            else:
                self.add_result(stage, CheckResult("Імпорт default_config", False, err))
        
        # 1.4 Перевірка JSON/YAML збереження
        if check_file_exists(config_dir / "default_config.py"):
            success, module, _ = try_import_module(config_dir / "default_config.py", "config_check")
            if success:
                has_save = check_function_exists(module, "save_config")
                has_load = check_function_exists(module, "load_config")
                self.add_result(stage, CheckResult("save_config()", has_save,
                    "Функція існує" if has_save else "Функція не знайдена"))
                self.add_result(stage, CheckResult("load_config()", has_load,
                    "Функція існує" if has_load else "Функція не знайдена"))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 2: schemas/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_2_schemas(self) -> List[CheckResult]:
        """Етап 2: Структури даних"""
        stage = "2_schemas"
        schemas_dir = self.root / "schemas"
        
        if not check_dir_exists(schemas_dir):
            self.add_result(stage, CheckResult("schemas/", False, "Папка schemas/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("schemas/", True, "Папка існує"))
        
        # 2.1 Перевірка файлів
        required_files = {
            "case_record.py": ["Symptom", "CaseRecord", "Patient"],
            "x_patient_sym.py": ["PatientVector"],
            "som_result.py": ["SOMResult", "UnitMembership"],
            "candidate_diagnoses.py": ["CandidateDiagnoses"],
            "nn_input_payload.py": ["NNInputPayload"],
            "iteration_state.py": ["IterationState"],
        }
        
        for filename, classes in required_files.items():
            filepath = schemas_dir / filename
            if not check_file_exists(filepath):
                self.add_result(stage, CheckResult(f"schemas/{filename}", False, "Файл відсутній"))
                continue
            
            self.add_result(stage, CheckResult(f"schemas/{filename}", True, "Файл існує"))
            
            # Перевірка класів
            success, module, err = try_import_module(filepath, f"schemas_{filename}")
            if success:
                for cls_name in classes:
                    exists = check_class_exists(module, cls_name)
                    self.add_result(stage, CheckResult(f"  {cls_name}", exists,
                        "Клас визначено" if exists else "Клас не знайдено"))
            else:
                self.add_result(stage, CheckResult(f"  Імпорт {filename}", False, err))
        
        # 2.2 Перевірка Pydantic або dataclass
        self.add_result(stage, CheckResult("Pydantic/dataclass", True, 
            "Перевірте вручну: чи використовується Pydantic або @dataclass", details="manual"))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 3: encoding/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_3_encoding(self) -> List[CheckResult]:
        """Етап 3: Векторизація"""
        stage = "3_encoding"
        encoding_dir = self.root / "encoding"
        
        if not check_dir_exists(encoding_dir):
            self.add_result(stage, CheckResult("encoding/", False, "Папка encoding/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("encoding/", True, "Папка існує"))
        
        # 3.1 Перевірка файлів
        required_files = {
            "symptom_vocabulary.py": ["SymptomVocabulary"],
            "symptom_encoder.py": ["SymptomEncoder"],
            "disease_encoder.py": ["DiseaseEncoder"],
            "patient_encoder.py": ["PatientEncoder"],
        }
        
        vocab_module = None
        disease_encoder_module = None
        
        for filename, classes in required_files.items():
            filepath = encoding_dir / filename
            if not check_file_exists(filepath):
                self.add_result(stage, CheckResult(f"encoding/{filename}", False, "Файл відсутній"))
                continue
            
            self.add_result(stage, CheckResult(f"encoding/{filename}", True, "Файл існує"))
            
            success, module, err = try_import_module(filepath, f"encoding_{filename}")
            if success:
                for cls_name in classes:
                    exists = check_class_exists(module, cls_name)
                    self.add_result(stage, CheckResult(f"  {cls_name}", exists,
                        "Клас визначено" if exists else "Клас не знайдено"))
                
                if filename == "symptom_vocabulary.py":
                    vocab_module = module
                elif filename == "disease_encoder.py":
                    disease_encoder_module = module
            else:
                self.add_result(stage, CheckResult(f"  Імпорт {filename}", False, err))
        
        # 3.2 Перевірка словника симптомів (461)
        data_dir = self.root / "data"
        vocab_file = data_dir / "symptom_vocabulary.json"
        
        if check_file_exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                symptom_count = len(vocab_data) if isinstance(vocab_data, (list, dict)) else 0
                
                # Якщо словник — dict з ключем "symptoms" або подібним
                if isinstance(vocab_data, dict):
                    if "symptoms" in vocab_data:
                        symptom_count = len(vocab_data["symptoms"])
                    elif "symptom_to_index" in vocab_data:
                        symptom_count = len(vocab_data["symptom_to_index"])
                
                is_461 = symptom_count == 461
                self.add_result(stage, CheckResult("Словник симптомів", is_461,
                    f"Знайдено {symptom_count} симптомів (очікується 461)"))
            except Exception as e:
                self.add_result(stage, CheckResult("Словник симптомів", False, f"Помилка читання: {e}"))
        else:
            self.add_result(stage, CheckResult("data/symptom_vocabulary.json", False, "Файл відсутній"))
        
        # 3.3 Перевірка бази діагнозів
        disease_file = data_dir / "unified_disease_symptom_data_full.json"
        if check_file_exists(disease_file):
            try:
                with open(disease_file, 'r', encoding='utf-8') as f:
                    disease_data = json.load(f)
                disease_count = len(disease_data) if isinstance(disease_data, (list, dict)) else 0
                
                if isinstance(disease_data, dict):
                    disease_count = len(disease_data.keys())
                
                is_842 = disease_count == 842
                self.add_result(stage, CheckResult("База діагнозів", is_842,
                    f"Знайдено {disease_count} діагнозів (очікується 842)"))
            except Exception as e:
                self.add_result(stage, CheckResult("База діагнозів", False, f"Помилка читання: {e}"))
        else:
            self.add_result(stage, CheckResult("data/unified_disease_symptom_data_full.json", False, 
                "Файл відсутній"))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 4: som/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_4_som(self) -> List[CheckResult]:
        """Етап 4: Self-Organizing Map"""
        stage = "4_som"
        som_dir = self.root / "som"
        
        if not check_dir_exists(som_dir):
            self.add_result(stage, CheckResult("som/", False, "Папка som/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("som/", True, "Папка існує"))
        
        # 4.1 Перевірка файлів
        required_files = {
            "som_model.py": ["SOM"],
            "som_training.py": ["SOMTrainer"],
            "som_index.py": ["SOMIndex"],
            "som_projection.py": ["SOMProjector"],
        }
        
        for filename, classes in required_files.items():
            filepath = som_dir / filename
            if not check_file_exists(filepath):
                self.add_result(stage, CheckResult(f"som/{filename}", False, "Файл відсутній"))
                continue
            
            self.add_result(stage, CheckResult(f"som/{filename}", True, "Файл існує"))
            
            success, module, err = try_import_module(filepath, f"som_{filename}")
            if success:
                for cls_name in classes:
                    exists = check_class_exists(module, cls_name)
                    self.add_result(stage, CheckResult(f"  {cls_name}", exists,
                        "Клас визначено" if exists else "Клас не знайдено"))
            else:
                self.add_result(stage, CheckResult(f"  Імпорт {filename}", False, err))
        
        # 4.2 Перевірка збереженої моделі SOM
        data_dir = self.root / "data"
        models_dir = self.root / "models"
        
        som_model_files = list(data_dir.glob("*som*.pkl")) + list(data_dir.glob("*som*.joblib"))
        som_model_files += list(models_dir.glob("*som*.pkl")) if models_dir.exists() else []
        som_model_files += list(models_dir.glob("*som*.joblib")) if models_dir.exists() else []
        
        if som_model_files:
            self.add_result(stage, CheckResult("Збережена SOM модель", True, 
                f"Знайдено: {[f.name for f in som_model_files[:3]]}"))
        else:
            self.add_result(stage, CheckResult("Збережена SOM модель", False, 
                "Не знайдено .pkl/.joblib файлів SOM"))
        
        # 4.3 Перевірка індексу юніт → діагнози
        index_file = data_dir / "som_disease_index.json"
        if check_file_exists(index_file):
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                unit_count = len(index_data) if isinstance(index_data, dict) else 0
                self.add_result(stage, CheckResult("som_disease_index.json", True, 
                    f"Знайдено {unit_count} юнітів"))
            except Exception as e:
                self.add_result(stage, CheckResult("som_disease_index.json", False, f"Помилка: {e}"))
        else:
            self.add_result(stage, CheckResult("data/som_disease_index.json", False, "Файл відсутній"))
        
        # 4.4 Метрики якості (QE, TE)
        self.add_result(stage, CheckResult("QE < 0.5, TE < 0.2", True, 
            "Перевірте вручну після запуску тестів", details="manual"))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 5: optimization/som_tuner.py
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_5_som_tuner(self) -> List[CheckResult]:
        """Етап 5: Тюнінг SOM"""
        stage = "5_som_tuner"
        opt_dir = self.root / "optimization"
        
        if not check_dir_exists(opt_dir):
            self.add_result(stage, CheckResult("optimization/", False, "Папка optimization/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("optimization/", True, "Папка існує"))
        
        filepath = opt_dir / "som_tuner.py"
        if not check_file_exists(filepath):
            self.add_result(stage, CheckResult("som_tuner.py", False, "Файл відсутній"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("som_tuner.py", True, "Файл існує"))
        
        success, module, err = try_import_module(filepath, "som_tuner")
        if success:
            has_tuner = check_class_exists(module, "SOMTuner")
            self.add_result(stage, CheckResult("  SOMTuner", has_tuner,
                "Клас визначено" if has_tuner else "Клас не знайдено"))
            
            if has_tuner:
                tuner_cls = getattr(module, "SOMTuner")
                has_tune = hasattr(tuner_cls, "tune")
                self.add_result(stage, CheckResult("  SOMTuner.tune()", has_tune,
                    "Метод існує" if has_tune else "Метод не знайдено"))
        else:
            self.add_result(stage, CheckResult("  Імпорт", False, err))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 6: candidate_selector/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_6_candidate_selector(self) -> List[CheckResult]:
        """Етап 6: Відбір кандидатів"""
        stage = "6_candidate_selector"
        selector_dir = self.root / "candidate_selector"
        
        if not check_dir_exists(selector_dir):
            self.add_result(stage, CheckResult("candidate_selector/", False, 
                "Папка candidate_selector/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("candidate_selector/", True, "Папка існує"))
        
        required_files = {
            "membership.py": ["MembershipCalculator"],
            "selector.py": ["CandidateSelector"],
            "guarantees.py": ["RecallValidator"],
        }
        
        selector_module = None
        
        for filename, classes in required_files.items():
            filepath = selector_dir / filename
            if not check_file_exists(filepath):
                self.add_result(stage, CheckResult(f"candidate_selector/{filename}", False, 
                    "Файл відсутній"))
                continue
            
            self.add_result(stage, CheckResult(f"candidate_selector/{filename}", True, "Файл існує"))
            
            success, module, err = try_import_module(filepath, f"selector_{filename}")
            if success:
                for cls_name in classes:
                    exists = check_class_exists(module, cls_name)
                    self.add_result(stage, CheckResult(f"  {cls_name}", exists,
                        "Клас визначено" if exists else "Клас не знайдено"))
                
                if filename == "selector.py":
                    selector_module = module
            else:
                self.add_result(stage, CheckResult(f"  Імпорт {filename}", False, err))
        
        # Перевірка політик відбору
        if selector_module and check_class_exists(selector_module, "CandidateSelector"):
            selector_cls = getattr(selector_module, "CandidateSelector")
            
            # Читаємо код файлу для перевірки політик
            selector_file = selector_dir / "selector.py"
            try:
                with open(selector_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                policies = ["top_k", "threshold", "cumulative_mass", "combined"]
                found_policies = [p for p in policies if p in code.lower()]
                
                self.add_result(stage, CheckResult("Політики відбору", len(found_policies) >= 3,
                    f"Знайдено: {found_policies}"))
            except:
                pass
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 7: optimization/selector_tuner.py
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_7_selector_tuner(self) -> List[CheckResult]:
        """Етап 7: Тюнінг Selector"""
        stage = "7_selector_tuner"
        opt_dir = self.root / "optimization"
        
        filepath = opt_dir / "selector_tuner.py"
        if not check_file_exists(filepath):
            self.add_result(stage, CheckResult("selector_tuner.py", False, "Файл відсутній"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("selector_tuner.py", True, "Файл існує"))
        
        success, module, err = try_import_module(filepath, "selector_tuner")
        if success:
            has_tuner = check_class_exists(module, "SelectorTuner")
            self.add_result(stage, CheckResult("  SelectorTuner", has_tuner,
                "Клас визначено" if has_tuner else "Клас не знайдено"))
        else:
            self.add_result(stage, CheckResult("  Імпорт", False, err))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 8: pseudo_generation/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_8_pseudo_generation(self) -> List[CheckResult]:
        """Етап 8: Генерація псевдопацієнтів"""
        stage = "8_pseudo_generation"
        pseudo_dir = self.root / "pseudo_generation"
        
        if not check_dir_exists(pseudo_dir):
            self.add_result(stage, CheckResult("pseudo_generation/", False, 
                "Папка pseudo_generation/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("pseudo_generation/", True, "Папка існує"))
        
        required_files = {
            "symptom_dropout.py": [],
            "noise_injection.py": [],
            "comorbidity_mixer.py": [],
            "iterative_generator.py": [],
        }
        
        for filename in required_files.keys():
            filepath = pseudo_dir / filename
            exists = check_file_exists(filepath)
            self.add_result(stage, CheckResult(f"pseudo_generation/{filename}", exists,
                "Файл існує" if exists else "Файл відсутній"))
        
        # Перевірка PseudoPatientGenerator
        for py_file in pseudo_dir.glob("*.py"):
            success, module, err = try_import_module(py_file, f"pseudo_{py_file.stem}")
            if success and check_class_exists(module, "PseudoPatientGenerator"):
                self.add_result(stage, CheckResult("PseudoPatientGenerator", True, 
                    f"Знайдено в {py_file.name}"))
                break
        else:
            self.add_result(stage, CheckResult("PseudoPatientGenerator", False, 
                "Клас не знайдено в жодному файлі"))
        
        # Перевірка згенерованих даних
        data_dir = self.root / "data"
        pseudo_files = list(data_dir.glob("*pseudo*.json")) + list(data_dir.glob("*pseudo*.pkl"))
        pseudo_files += list(data_dir.glob("*training*.json")) + list(data_dir.glob("*training*.pkl"))
        
        if pseudo_files:
            self.add_result(stage, CheckResult("Згенеровані дані", True, 
                f"Знайдено: {[f.name for f in pseudo_files[:3]]}"))
        else:
            self.add_result(stage, CheckResult("Згенеровані дані", False, 
                "Не знайдено файлів псевдокейсів", details="optional"))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 9: optimization/generation_tuner.py
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_9_generation_tuner(self) -> List[CheckResult]:
        """Етап 9: Тюнінг генерації"""
        stage = "9_generation_tuner"
        opt_dir = self.root / "optimization"
        
        filepath = opt_dir / "generation_tuner.py"
        if not check_file_exists(filepath):
            self.add_result(stage, CheckResult("generation_tuner.py", False, "Файл відсутній"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("generation_tuner.py", True, "Файл існує"))
        
        success, module, err = try_import_module(filepath, "generation_tuner")
        if success:
            has_tuner = check_class_exists(module, "GenerationTuner")
            self.add_result(stage, CheckResult("  GenerationTuner", has_tuner,
                "Клас визначено" if has_tuner else "Клас не знайдено"))
        else:
            self.add_result(stage, CheckResult("  Імпорт", False, err))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ЕТАП 10: multilabel_nn/
    # ───────────────────────────────────────────────────────────────────────────
    def verify_stage_10_multilabel_nn(self) -> List[CheckResult]:
        """Етап 10: Multilabel Neural Network"""
        stage = "10_multilabel_nn"
        nn_dir = self.root / "multilabel_nn"
        
        if not check_dir_exists(nn_dir):
            self.add_result(stage, CheckResult("multilabel_nn/", False, 
                "Папка multilabel_nn/ не існує"))
            return self.results.get(stage, [])
        
        self.add_result(stage, CheckResult("multilabel_nn/", True, "Папка існує"))
        
        required_files = {
            "model.py": ["TwoBranchNN", "SimpleMLP"],
            "training.py": ["NNTrainer"],
            "inference.py": ["NNInference"],
            "metrics.py": [],
        }
        
        for filename, classes in required_files.items():
            filepath = nn_dir / filename
            if not check_file_exists(filepath):
                self.add_result(stage, CheckResult(f"multilabel_nn/{filename}", False, 
                    "Файл відсутній"))
                continue
            
            self.add_result(stage, CheckResult(f"multilabel_nn/{filename}", True, "Файл існує"))
            
            if classes:
                success, module, err = try_import_module(filepath, f"nn_{filename}")
                if success:
                    for cls_name in classes:
                        exists = check_class_exists(module, cls_name)
                        # TwoBranchNN або SimpleMLP — достатньо одного
                        if cls_name in ["TwoBranchNN", "SimpleMLP"]:
                            if exists:
                                self.add_result(stage, CheckResult(f"  {cls_name}", True, 
                                    "Клас визначено"))
                        else:
                            self.add_result(stage, CheckResult(f"  {cls_name}", exists,
                                "Клас визначено" if exists else "Клас не знайдено"))
                else:
                    self.add_result(stage, CheckResult(f"  Імпорт {filename}", False, err))
        
        # Перевірка збереженої моделі
        models_dir = self.root / "models"
        data_dir = self.root / "data"
        
        nn_model_files = []
        for d in [models_dir, data_dir, nn_dir]:
            if d.exists():
                nn_model_files += list(d.glob("*.pt"))
                nn_model_files += list(d.glob("*.pth"))
                nn_model_files += list(d.glob("*nn*.pkl"))
        
        if nn_model_files:
            self.add_result(stage, CheckResult("Збережена NN модель", True, 
                f"Знайдено: {[f.name for f in nn_model_files[:3]]}"))
        else:
            self.add_result(stage, CheckResult("Збережена NN модель", False, 
                "Не знайдено .pt/.pth файлів моделі"))
        
        # Метрики
        self.add_result(stage, CheckResult("Recall@5 ≥ 0.85, Recall@10 ≥ 0.92", True, 
            "Перевірте вручну після запуску тестів", details="manual"))
        
        return self.results.get(stage, [])

    # ───────────────────────────────────────────────────────────────────────────
    # ГОЛОВНИЙ МЕТОД
    # ───────────────────────────────────────────────────────────────────────────
    def verify_all(self) -> Dict[str, List[CheckResult]]:
        """Запустити всі перевірки"""
        print(header("Dr.Case — Перевірка етапів 1-10"))
        print(f"\n{info(f'Шлях до проекту: {self.root}')}\n")
        
        stages = [
            ("Етап 1: config/", self.verify_stage_1_config),
            ("Етап 2: schemas/", self.verify_stage_2_schemas),
            ("Етап 3: encoding/", self.verify_stage_3_encoding),
            ("Етап 4: som/", self.verify_stage_4_som),
            ("Етап 5: optimization/som_tuner.py", self.verify_stage_5_som_tuner),
            ("Етап 6: candidate_selector/", self.verify_stage_6_candidate_selector),
            ("Етап 7: optimization/selector_tuner.py", self.verify_stage_7_selector_tuner),
            ("Етап 8: pseudo_generation/", self.verify_stage_8_pseudo_generation),
            ("Етап 9: optimization/generation_tuner.py", self.verify_stage_9_generation_tuner),
            ("Етап 10: multilabel_nn/", self.verify_stage_10_multilabel_nn),
        ]
        
        for stage_name, verify_func in stages:
            print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {stage_name}{Colors.END}")
            print("-" * 50)
            
            try:
                results = verify_func()
                for r in results:
                    if r.passed:
                        print(ok(f"{r.name}: {r.message}"))
                    elif r.details == "optional":
                        print(warn(f"{r.name}: {r.message}"))
                    elif r.details == "manual":
                        print(info(f"{r.name}: {r.message}"))
                    else:
                        print(fail(f"{r.name}: {r.message}"))
            except Exception as e:
                print(fail(f"Помилка перевірки: {e}"))
                traceback.print_exc()
        
        return self.results

    def print_summary(self):
        """Друк підсумку"""
        print(header("ПІДСУМОК"))
        
        total_passed = 0
        total_failed = 0
        total_warnings = 0
        
        for stage, results in self.results.items():
            passed = sum(1 for r in results if r.passed and r.details != "manual")
            failed = sum(1 for r in results if not r.passed and r.details not in ["optional", "manual"])
            warnings = sum(1 for r in results if not r.passed and r.details == "optional")
            manual = sum(1 for r in results if r.details == "manual")
            
            total_passed += passed
            total_failed += failed
            total_warnings += warnings
            
            status = "✓" if failed == 0 else "✗"
            color = Colors.GREEN if failed == 0 else Colors.RED
            
            print(f"{color}{status}{Colors.END} {stage}: {passed} passed, {failed} failed, {warnings} warnings, {manual} manual")
        
        print("-" * 50)
        print(f"Всього: {Colors.GREEN}{total_passed} passed{Colors.END}, "
              f"{Colors.RED}{total_failed} failed{Colors.END}, "
              f"{Colors.YELLOW}{total_warnings} warnings{Colors.END}")
        
        if total_failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Всі обов'язкові перевірки пройдено!{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ Є проблеми, що потребують виправлення.{Colors.END}")


# ═══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДУ
# ═══════════════════════════════════════════════════════════════════════════════

def find_project_root() -> Optional[Path]:
    """Пошук кореня проекту dr_case"""
    candidates = [
        Path.cwd() / "dr_case",
        Path.cwd(),
        Path(__file__).parent / "dr_case",
        Path(__file__).parent,
    ]
    
    for candidate in candidates:
        if (candidate / "config").exists() or (candidate / "schemas").exists():
            return candidate
    
    return None


def main():
    # Визначення шляху
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = find_project_root()
    
    if project_path is None:
        print(fail("Не вдалося знайти проект dr_case"))
        print(info("Використання: python verify_stages.py [шлях_до_dr_case]"))
        sys.exit(1)
    
    if not project_path.exists():
        print(fail(f"Шлях не існує: {project_path}"))
        sys.exit(1)
    
    # Запуск перевірки
    verifier = StageVerifier(project_path)
    verifier.verify_all()
    verifier.print_summary()


if __name__ == "__main__":
    main()
