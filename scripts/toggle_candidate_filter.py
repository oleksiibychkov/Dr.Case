#!/usr/bin/env python3
"""
Dr.Case — Налаштування фільтрації по кандидатах

Якщо Candidate Recall ≥95%, можна увімкнути фільтрацію для швидшого inference.
Якщо Candidate Recall <95%, краще залишити bypass.

Запуск:
    python scripts/toggle_candidate_filter.py --enable
    python scripts/toggle_candidate_filter.py --disable
    python scripts/toggle_candidate_filter.py --status
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
cycle_controller_path = project_root / "dr_case" / "diagnosis_cycle" / "cycle_controller.py"


def get_status():
    """Перевірити поточний стан фільтрації"""
    with open(cycle_controller_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "BYPASS:" in content and "filtered_hyp = hypotheses" in content:
        return "disabled"
    elif "filtered_hyp = {d: p for d, p in hypotheses.items() if d in candidates}" in content:
        return "enabled"
    else:
        return "unknown"


def enable_filter():
    """Увімкнути фільтрацію по кандидатах"""
    with open(cycle_controller_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Шукаємо блок з BYPASS
    old_block = '''        # 4. NN → гіпотези
        hypotheses = self._predict_hypotheses(symptom_vector, membership)
        
        # BYPASS: Не фільтруємо по кандидатах, бо SOM має високий QE (2.25)
        # NN показує 93% Recall@5 без фільтрації
        # TODO: Перенавчити SOM для кращої кластеризації, тоді повернути фільтрацію
        filtered_hyp = hypotheses  # Використовуємо всі гіпотези від NN
        
        # Нормалізуємо (softmax вже нормалізований)
        total = sum(filtered_hyp.values())'''
    
    new_block = '''        # 4. NN → гіпотези
        hypotheses = self._predict_hypotheses(symptom_vector, membership)
        
        # Фільтруємо по кандидатах і нормалізуємо
        filtered_hyp = {d: p for d, p in hypotheses.items() if d in candidates}
        total = sum(filtered_hyp.values())'''
    
    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(cycle_controller_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Фільтрація по кандидатах УВІМКНЕНА")
        return True
    else:
        print("⚠️ Не знайдено блок для заміни. Можливо фільтрація вже увімкнена.")
        return False


def disable_filter():
    """Вимкнути фільтрацію по кандидатах (bypass)"""
    with open(cycle_controller_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_block = '''        # 4. NN → гіпотези
        hypotheses = self._predict_hypotheses(symptom_vector, membership)
        
        # Фільтруємо по кандидатах і нормалізуємо
        filtered_hyp = {d: p for d, p in hypotheses.items() if d in candidates}
        total = sum(filtered_hyp.values())'''
    
    new_block = '''        # 4. NN → гіпотези
        hypotheses = self._predict_hypotheses(symptom_vector, membership)
        
        # BYPASS: Не фільтруємо по кандидатах, бо SOM має високий QE (2.25)
        # NN показує 93% Recall@5 без фільтрації
        # TODO: Перенавчити SOM для кращої кластеризації, тоді повернути фільтрацію
        filtered_hyp = hypotheses  # Використовуємо всі гіпотези від NN
        
        # Нормалізуємо (softmax вже нормалізований)
        total = sum(filtered_hyp.values())'''
    
    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(cycle_controller_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Фільтрація по кандидатах ВИМКНЕНА (bypass)")
        return True
    else:
        print("⚠️ Не знайдено блок для заміни. Можливо bypass вже активний.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Налаштування фільтрації по кандидатах')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--enable', action='store_true', help='Увімкнути фільтрацію')
    group.add_argument('--disable', action='store_true', help='Вимкнути фільтрацію (bypass)')
    group.add_argument('--status', action='store_true', help='Показати поточний стан')
    
    args = parser.parse_args()
    
    if args.status:
        status = get_status()
        print(f"📊 Поточний стан фільтрації: {status.upper()}")
        if status == "disabled":
            print("   NN ранжує всі 844 хвороби (повільніше, але надійніше)")
        elif status == "enabled":
            print("   NN ранжує тільки кандидатів від SOM (швидше)")
    elif args.enable:
        enable_filter()
    elif args.disable:
        disable_filter()


if __name__ == "__main__":
    main()
