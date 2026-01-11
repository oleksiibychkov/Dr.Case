"""Тест модуля config"""
from dr_case.config import DrCaseConfig, get_default_config


def demo():
    print("=" * 50)
    print("Dr.Case — Тест конфігурації")
    print("=" * 50)
    
    config = get_default_config()
    
    print(f"Версія: {config.version}")
    print(f"Діагнозів: {config.n_diseases}")
    print(f"Симптомів: {config.n_symptoms}")
    print(f"SOM grid: {config.som.grid_height}x{config.som.grid_width}")
    print(f"Candidate α: {config.candidate_selector.alpha}")
    print(f"Learning rate: {config.nn_training.learning_rate}")
    
    print("=" * 50)
    print("✅ Успішно!")


if __name__ == "__main__":
    demo()