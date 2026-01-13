"""
Стандартизація до lowercase

Конвертує:
1. База даних JSON: всі назви діагнозів та симптомів → lowercase
2. Перенавчує SOM
3. Перенавчує NN

Після цього вся система використовує єдиний формат.
"""

import json
import sys
from pathlib import Path

# Додаємо корінь проекту до path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def convert_database_to_lowercase(input_path: str, output_path: str) -> dict:
    """
    Конвертує базу даних в lowercase.

    Структура бази:
    {
        "Disease Name": {
            "symptoms": ["Symptom1", "Symptom2", ...],
            "symptom_frequency": {"Symptom1": 0.8, "Symptom2": 0.5, ...},
            "description": "...",
            ...
        }
    }

    Після конвертації:
    {
        "disease name": {
            "symptoms": ["symptom1", "symptom2", ...],
            "symptom_frequency": {"symptom1": 0.8, "symptom2": 0.5, ...},
            ...
        }
    }
    """
    print(f"[1] Завантаження бази з {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        database = json.load(f)

    print(f"    Діагнозів: {len(database)}")

    # Збираємо статистику
    all_symptoms_before = set()
    for disease_data in database.values():
        all_symptoms_before.update(disease_data.get('symptoms', []))
    print(f"    Унікальних симптомів (до): {len(all_symptoms_before)}")

    # Конвертуємо
    print(f"\n[2] Конвертація в lowercase...")
    new_database = {}

    for disease_name, disease_data in database.items():
        # Назва діагнозу → lowercase
        new_name = disease_name.lower()

        # Копіюємо дані
        new_data = {}
        for key, value in disease_data.items():
            if key == 'symptoms':
                # Симптоми → lowercase
                new_data[key] = [s.lower() for s in value]
            elif key == 'symptom_frequency':
                # КЛЮЧІ symptom_frequency → lowercase
                new_data[key] = {k.lower(): v for k, v in value.items()}
            elif key == 'treatments':
                # Лікування → lowercase
                new_data[key] = [t.lower() for t in value]
            elif key == 'tests':
                # Тести → lowercase
                new_data[key] = [t.lower() for t in value]
            else:
                new_data[key] = value

        # Перевірка на дублікати
        if new_name in new_database:
            print(f"    УВАГА: Дублікат діагнозу '{new_name}' (з '{disease_name}')")
            # Об'єднуємо симптоми
            existing_symptoms = set(new_database[new_name].get('symptoms', []))
            new_symptoms = set(new_data.get('symptoms', []))
            merged = list(existing_symptoms | new_symptoms)
            new_database[new_name]['symptoms'] = merged
        else:
            new_database[new_name] = new_data

    # Статистика після
    all_symptoms_after = set()
    all_freq_keys_after = set()
    for disease_data in new_database.values():
        all_symptoms_after.update(disease_data.get('symptoms', []))
        all_freq_keys_after.update(disease_data.get('symptom_frequency', {}).keys())

    print(f"    Діагнозів (після): {len(new_database)}")
    print(f"    Унікальних симптомів (після): {len(all_symptoms_after)}")
    print(f"    Унікальних freq ключів (після): {len(all_freq_keys_after)}")

    # Зберігаємо
    print(f"\n[3] Збереження в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_database, f, ensure_ascii=False, indent=2)

    print(f"    Готово!")

    return new_database


def train_som_lowercase(database_path: str, output_path: str):
    """Навчити SOM на lowercase базі"""
    print(f"\n[4] Навчання SOM...")

    from dr_case.som import SOMTrainer
    from dr_case.config import SOMConfig

    # 20x20
    config = SOMConfig(
        grid_height=20,
        grid_width=20,
        input_dim=460,
        epochs=1000,
    )

    trainer = SOMTrainer(config)
    som, metrics = trainer.train_from_database(database_path)

    print(f"    QE: {metrics['qe']:.4f}")
    print(f"    TE: {metrics['te']:.4f}")
    print(f"    Fill ratio: {metrics['fill_ratio']:.2%}")

    trainer.save_model(output_path)

    return som


def train_nn_lowercase(database_path: str, som_path: str, output_path: str):
    """Навчити NN на lowercase базі"""
    print(f"\n[5] Навчання NN...")

    from dr_case.neural_network.trainer_bmu import BMUTrainer, BMUTrainingConfig

    config = BMUTrainingConfig(
        samples_per_disease=100,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
    )

    trainer = BMUTrainer(database_path, som_path, config)
    metrics = trainer.train(verbose=True)
    trainer.save(output_path)

    print(f"    Val Accuracy: {metrics['val_acc']:.1f}%")
    print(f"    Val Top-5: {metrics['val_top5']:.1f}%")
    print(f"    Збережено в {output_path}")

    return metrics


def verify_consistency(database_path: str, som_path: str, nn_path: str):
    """Перевірити консистентність після конвертації"""
    import torch
    from dr_case.som import SOMModel

    print(f"\n[6] Перевірка консистентності...")

    # База
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)

    db_diseases = set(database.keys())
    db_symptoms = set()
    db_freq_keys = set()
    for d in database.values():
        db_symptoms.update(d.get('symptoms', []))
        db_freq_keys.update(d.get('symptom_frequency', {}).keys())

    # SOM
    som = SOMModel.load(som_path)
    som_diseases = set(som._disease_names)

    # NN
    checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
    nn_diseases = set(checkpoint['disease_names'])
    nn_symptoms = set(checkpoint['symptom_names'])

    # Перевірки
    print(f"\n    База даних:")
    print(f"      Діагнозів: {len(db_diseases)}")
    print(f"      Симптомів: {len(db_symptoms)}")
    print(f"      Freq ключів: {len(db_freq_keys)}")

    print(f"\n    SOM:")
    print(f"      Діагнозів: {len(som_diseases)}")

    print(f"\n    NN:")
    print(f"      Діагнозів: {len(nn_diseases)}")
    print(f"      Симптомів: {len(nn_symptoms)}")

    # Перевірка lowercase
    db_upper = [d for d in db_diseases if d != d.lower()]
    sym_upper = [s for s in db_symptoms if s != s.lower()]
    freq_upper = [k for k in db_freq_keys if k != k.lower()]
    som_upper = [d for d in som_diseases if d != d.lower()]
    nn_upper = [d for d in nn_diseases if d != d.lower()]
    nn_sym_upper = [s for s in nn_symptoms if s != s.lower()]

    if db_upper:
        print(f"\n    ⚠️ База: НЕ lowercase діагнози: {db_upper[:5]}...")
    else:
        print(f"\n    ✅ База: всі діагнози lowercase")

    if sym_upper:
        print(f"    ⚠️ База: НЕ lowercase симптоми: {sym_upper[:5]}...")
    else:
        print(f"    ✅ База: всі симптоми lowercase")

    if freq_upper:
        print(f"    ⚠️ База: НЕ lowercase freq ключі: {freq_upper[:5]}...")
    else:
        print(f"    ✅ База: всі freq ключі lowercase")

    if som_upper:
        print(f"    ⚠️ SOM: НЕ lowercase діагнози: {som_upper[:5]}...")
    else:
        print(f"    ✅ SOM: всі діагнози lowercase")

    if nn_upper:
        print(f"    ⚠️ NN: НЕ lowercase діагнози: {nn_upper[:5]}...")
    else:
        print(f"    ✅ NN: всі діагнози lowercase")

    if nn_sym_upper:
        print(f"    ⚠️ NN: НЕ lowercase симптоми: {nn_sym_upper[:5]}...")
    else:
        print(f"    ✅ NN: всі симптоми lowercase")

    # Перевірка відповідності
    if db_diseases == som_diseases == nn_diseases:
        print(f"\n    ✅ Всі діагнози співпадають!")
    else:
        diff_som = db_diseases - som_diseases
        diff_nn = db_diseases - nn_diseases
        if diff_som:
            print(f"\n    ⚠️ В базі є, в SOM немає: {list(diff_som)[:5]}...")
        if diff_nn:
            print(f"    ⚠️ В базі є, в NN немає: {list(diff_nn)[:5]}...")


def main():
    print("=" * 60)
    print("СТАНДАРТИЗАЦІЯ ДО LOWERCASE")
    print("=" * 60)

    # Шляхи
    input_db = 'data/unified_disease_symptom_merged.json'
    output_db = 'data/database_lowercase.json'
    som_path = 'models/som_lowercase.pkl'
    nn_path = 'models/nn_lowercase.pt'

    # 1. Конвертація бази
    database = convert_database_to_lowercase(input_db, output_db)

    # 2. Навчання SOM
    som = train_som_lowercase(output_db, som_path)

    # 3. Навчання NN
    metrics = train_nn_lowercase(output_db, som_path, nn_path)

    # 4. Перевірка
    verify_consistency(output_db, som_path, nn_path)

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"""
Нові файли:
  - {output_db}
  - {som_path}
  - {nn_path}

Для використання оновіть шляхи в коді:
  database_path = '{output_db}'
  som_path = '{som_path}'
  nn_path = '{nn_path}'
""")


if __name__ == '__main__':
    main()
