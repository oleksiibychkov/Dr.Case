"""
Виправлення бази даних - ВСІ поля в lowercase
"""

import json


def fix_database_lowercase(input_path: str, output_path: str):
    """Конвертувати ВСІ поля бази в lowercase"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    
    fixed_db = {}
    
    for disease_name, data in db.items():
        # Назва хвороби → lowercase
        disease_lower = disease_name.lower()
        
        fixed_data = {}
        
        for key, value in data.items():
            if key == 'symptoms':
                # Список симптомів → lowercase
                fixed_data['symptoms'] = [s.lower() for s in value]
                
            elif key == 'symptom_frequency':
                # Словник частот → ключі lowercase
                fixed_data['symptom_frequency'] = {
                    k.lower(): v for k, v in value.items()
                }
                
            elif key == 'treatments':
                # Лікування → lowercase
                fixed_data['treatments'] = [t.lower() for t in value]
                
            elif key == 'tests':
                # Тести → lowercase  
                fixed_data['tests'] = [t.lower() for t in value]
                
            else:
                # Інші поля залишаємо як є
                fixed_data[key] = value
        
        fixed_db[disease_lower] = fixed_data
    
    # Зберігаємо
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_db, f, ensure_ascii=False, indent=2)
    
    # Перевірка
    print(f"Diseases: {len(fixed_db)}")
    
    all_symptoms = set()
    all_freq_keys = set()
    
    for data in fixed_db.values():
        all_symptoms.update(data.get('symptoms', []))
        all_freq_keys.update(data.get('symptom_frequency', {}).keys())
    
    print(f"Unique symptoms: {len(all_symptoms)}")
    print(f"Unique freq keys: {len(all_freq_keys)}")
    
    # Перевірка консистентності
    symptoms_not_in_freq = all_symptoms - all_freq_keys
    freq_not_in_symptoms = all_freq_keys - all_symptoms
    
    print(f"\nSymptoms not in freq: {len(symptoms_not_in_freq)}")
    print(f"Freq not in symptoms: {len(freq_not_in_symptoms)}")
    
    # Приклад
    first_disease = list(fixed_db.keys())[0]
    first_data = fixed_db[first_disease]
    print(f"\nExample disease: {first_disease}")
    print(f"  symptoms: {first_data.get('symptoms', [])[:3]}")
    print(f"  freq keys: {list(first_data.get('symptom_frequency', {}).keys())[:3]}")
    
    return fixed_db


if __name__ == "__main__":
    fix_database_lowercase(
        'data/database_lowercase.json',
        'data/database_lowercase.json'  # Перезаписуємо
    )
    print("\n✅ Database fixed!")
