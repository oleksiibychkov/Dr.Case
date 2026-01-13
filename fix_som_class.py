"""
Фікс завантаження SOM моделі
Запустити перед test_new_architecture.py
"""
import pickle
import sys

# Імпортуємо клас з train_real_data
from train_real_data import SOMModelReal

# Реєструємо в __main__ для pickle
sys.modules['__main__'].SOMModelReal = SOMModelReal

# Тепер завантажуємо і пересберігаємо
print("Loading SOM...")
with open('models/som_real.pkl', 'rb') as f:
    som = pickle.load(f)

print(f"  Type: {type(som)}")
print(f"  Diseases: {len(som._disease_names)}")
print(f"  Symptoms: {len(som._symptom_names)}")

# Тепер імпортуємо оригінальний SOMModel і копіюємо дані
from dr_case.som.model import SOMModel

# Створюємо новий об'єкт з правильним класом
new_som = object.__new__(SOMModel)
new_som._som = som._som
new_som._disease_names = som._disease_names
new_som._symptom_names = som._symptom_names  
new_som._disease_index = som._disease_index
new_som._disease_vectors = som._disease_vectors
new_som._symptom_to_idx = som._symptom_to_idx
new_som._disease_to_idx = som._disease_to_idx

# Зберігаємо з правильним класом
print("Saving as SOMModel...")
with open('models/som_real.pkl', 'wb') as f:
    pickle.dump(new_som, f)

print("✅ Done! Now run: python test_new_architecture.py")
