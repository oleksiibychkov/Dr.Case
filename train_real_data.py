"""
Dr.Case — Повне тренування на реальних даних

1. SOM на БІНАРНИХ векторах симптомів
2. NN на БІНАРНИХ симптомах + BMU
3. Validation на згенерованих пацієнтах (за реальними частотами)
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from minisom import MiniSom
from typing import Dict, List, Tuple


# ============================================================
# SOM MODEL
# ============================================================

class SOMModelReal:
    """SOM модель з методами для діагностики"""
    
    def __init__(
        self,
        som: MiniSom,
        disease_names: list,
        symptom_names: list,
        disease_index: dict,
        disease_vectors: np.ndarray
    ):
        self._som = som
        self._disease_names = disease_names
        self._symptom_names = symptom_names
        self._disease_index = disease_index
        self._disease_vectors = disease_vectors
        
        self._symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}
        self._disease_to_idx = {d: i for i, d in enumerate(disease_names)}
    
    @classmethod
    def load(cls, path: str) -> 'SOMModelReal':
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_bmu(self, vector: np.ndarray) -> Tuple[int, int]:
        return self._som.winner(vector)
    
    def get_bmu_for_disease(self, disease_name: str) -> Tuple[int, int]:
        if disease_name not in self._disease_to_idx:
            return None
        idx = self._disease_to_idx[disease_name]
        return self.get_bmu(self._disease_vectors[idx])
    
    def get_diseases_at_bmu(self, bmu: Tuple[int, int]) -> list:
        key = f"{bmu[0]}_{bmu[1]}"
        return self._disease_index.get(key, [])
    
    def encode_symptoms(self, symptoms: list) -> np.ndarray:
        vector = np.zeros(len(self._symptom_names), dtype=np.float32)
        for symptom in symptoms:
            if symptom in self._symptom_to_idx:
                vector[self._symptom_to_idx[symptom]] = 1.0
        return vector
    
    def get_cluster_diseases(self, symptoms: list, radius: int = 1) -> list:
        vector = self.encode_symptoms(symptoms)
        bmu = self.get_bmu(vector)
        
        diseases = set()
        shape = self._som.get_weights().shape
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                i, j = bmu[0] + di, bmu[1] + dj
                if 0 <= i < shape[0] and 0 <= j < shape[1]:
                    diseases.update(self.get_diseases_at_bmu((i, j)))
        
        return list(diseases)
    
    def compute_quantization_error(self, data=None):
        if data is None:
            data = self._disease_vectors
        return self._som.quantization_error(data)
    
    def compute_topographic_error(self, data=None):
        if data is None:
            data = self._disease_vectors
        return self._som.topographic_error(data)
    
    def get_fill_ratio(self):
        filled = sum(1 for d in self._disease_index.values() if d)
        total = self._som.get_weights().shape[0] * self._som.get_weights().shape[1]
        return filled / total


# ============================================================
# NN MODEL
# ============================================================

class TwoBranchNN(nn.Module):
    """Двогілкова нейромережа"""
    
    def __init__(
        self,
        n_symptoms: int,
        n_diseases: int,
        som_dim: int = 2,
        symptom_hidden: List[int] = [256, 128],
        bmu_hidden: List[int] = [32, 16],
        combined_hidden: List[int] = [128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Symptom branch
        layers = []
        in_dim = n_symptoms
        for h in symptom_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.symptom_branch = nn.Sequential(*layers)
        
        # BMU branch
        layers = []
        in_dim = som_dim
        for h in bmu_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout * 0.5)])
            in_dim = h
        self.bmu_branch = nn.Sequential(*layers)
        
        # Combined
        layers = []
        in_dim = symptom_hidden[-1] + bmu_hidden[-1]
        for h in combined_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.combined = nn.Sequential(*layers)
        
        self.output = nn.Linear(combined_hidden[-1], n_diseases)
    
    def forward(self, symptoms, bmu):
        s = self.symptom_branch(symptoms)
        b = self.bmu_branch(bmu)
        c = torch.cat([s, b], dim=1)
        c = self.combined(c)
        return self.output(c)


# ============================================================
# TRAINING
# ============================================================

def load_database(path: str) -> Tuple[dict, list, list]:
    """Завантажити базу та отримати списки"""
    with open(path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    
    all_symptoms = set()
    for data in db.values():
        all_symptoms.update(data.get('symptom_frequency', {}).keys())
        all_symptoms.update(data.get('symptoms', []))
    
    symptom_names = sorted(all_symptoms)
    disease_names = sorted(db.keys())
    
    return db, disease_names, symptom_names


def encode_disease(db: dict, disease_name: str, symptom_to_idx: dict) -> np.ndarray:
    """Закодувати хворобу у БІНАРНИЙ вектор (симптом є = 1.0, немає = 0.0)"""
    data = db[disease_name]
    
    # Беремо симптоми з symptom_frequency (ключі) або symptoms (список)
    symptoms = set(data.get('symptom_frequency', {}).keys())
    symptoms.update(data.get('symptoms', []))
    
    vector = np.zeros(len(symptom_to_idx), dtype=np.float32)
    
    for symptom in symptoms:
        if symptom in symptom_to_idx:
            vector[symptom_to_idx[symptom]] = 1.0  # Бінарно: є симптом
    
    return vector


def train_som(
    db: dict,
    disease_names: list,
    symptom_names: list,
    grid_size: int = 20,
    epochs: int = 1000
) -> SOMModelReal:
    """Тренування SOM"""
    print("\n" + "=" * 60)
    print("ТРЕНУВАННЯ SOM")
    print("=" * 60)
    
    symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}
    
    # Encode all diseases
    vectors = np.array([
        encode_disease(db, d, symptom_to_idx) for d in disease_names
    ], dtype=np.float32)
    
    print(f"Data shape: {vectors.shape}")
    
    # Train SOM
    som = MiniSom(
        grid_size, grid_size,
        len(symptom_names),
        sigma=grid_size / 2,
        learning_rate=0.5,
        neighborhood_function='gaussian',
        random_seed=42
    )
    som.pca_weights_init(vectors)
    som.train(vectors, epochs, verbose=True)
    
    # Metrics
    qe = som.quantization_error(vectors)
    te = som.topographic_error(vectors)
    print(f"QE: {qe:.4f}")
    print(f"TE: {te:.4f}")
    
    # Build disease index
    disease_index = {}
    for i, disease_name in enumerate(disease_names):
        bmu = som.winner(vectors[i])
        key = f"{bmu[0]}_{bmu[1]}"
        if key not in disease_index:
            disease_index[key] = []
        disease_index[key].append(disease_name)
    
    # Fill ratio
    filled = sum(1 for d in disease_index.values() if d)
    total = grid_size * grid_size
    print(f"Fill ratio: {filled/total:.2%}")
    
    return SOMModelReal(
        som=som,
        disease_names=disease_names,
        symptom_names=symptom_names,
        disease_index=disease_index,
        disease_vectors=vectors
    )


def train_nn(
    db: dict,
    disease_names: list,
    symptom_names: list,
    som: SOMModelReal,
    epochs: int = 100,
    batch_size: int = 128,
    device: str = None
) -> Tuple[TwoBranchNN, dict]:
    """Тренування NN"""
    print("\n" + "=" * 60)
    print("ТРЕНУВАННЯ NN")
    print("=" * 60)
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    symptom_to_idx = {s: i for i, s in enumerate(symptom_names)}
    n_diseases = len(disease_names)
    n_symptoms = len(symptom_names)
    
    som_shape = som._som.get_weights().shape[:2]
    
    # Генеруємо TRAIN приклади за реальними частотами
    print("Generating training samples...")
    n_train_per_disease = 100
    X_sym_train_list = []
    X_bmu_train_list = []
    y_train_list = []
    
    for i, disease_name in enumerate(disease_names):
        data = db[disease_name]
        freq_dict = data.get('symptom_frequency', {})
        
        if not freq_dict:
            symptoms = data.get('symptoms', [])
            freq_dict = {s: 1.0 for s in symptoms}
        
        for _ in range(n_train_per_disease):
            patient_vector = np.zeros(n_symptoms, dtype=np.float32)
            
            for symptom, freq in freq_dict.items():
                if symptom in symptom_to_idx:
                    if freq > 1.0:
                        freq = freq / 100.0
                    if np.random.random() < freq:
                        patient_vector[symptom_to_idx[symptom]] = 1.0
            
            bmu = som.get_bmu(patient_vector)
            bmu_norm = [bmu[0] / som_shape[0], bmu[1] / som_shape[1]]
            
            X_sym_train_list.append(patient_vector)
            X_bmu_train_list.append(bmu_norm)
            y_train_list.append(i)
    
    X_symptoms = np.array(X_sym_train_list, dtype=np.float32)
    X_bmu = np.array(X_bmu_train_list, dtype=np.float32)
    y = np.array(y_train_list, dtype=np.int64)
    
    print(f"Training samples: {len(y)}")
    
    # Тренуємо на згенерованих даних
    train_dataset = TensorDataset(
        torch.FloatTensor(X_symptoms),
        torch.FloatTensor(X_bmu),
        torch.LongTensor(y)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Генеруємо валідаційні приклади (симуляція пацієнтів)
    print("Generating validation samples...")
    n_val_per_disease = 20
    X_sym_val_list = []
    X_bmu_val_list = []
    y_val_list = []
    
    for i, disease_name in enumerate(disease_names):
        # Беремо РЕАЛЬНІ ЧАСТОТИ з бази для генерації пацієнтів
        data = db[disease_name]
        freq_dict = data.get('symptom_frequency', {})
        
        # Якщо немає частот - використовуємо symptoms як 100%
        if not freq_dict:
            symptoms = data.get('symptoms', [])
            freq_dict = {s: 1.0 for s in symptoms}
        
        for _ in range(n_val_per_disease):
            # Симулюємо пацієнта: симптом присутній з ймовірністю = його частота
            patient_vector = np.zeros(n_symptoms, dtype=np.float32)
            
            for symptom, freq in freq_dict.items():
                if symptom in symptom_to_idx:
                    # Нормалізуємо частоту якщо > 1
                    if freq > 1.0:
                        freq = freq / 100.0
                    # Симптом присутній з ймовірністю = частота
                    if np.random.random() < freq:
                        patient_vector[symptom_to_idx[symptom]] = 1.0
            
            # BMU для цього пацієнта
            bmu = som.get_bmu(patient_vector)
            bmu_norm = [bmu[0] / som_shape[0], bmu[1] / som_shape[1]]
            
            X_sym_val_list.append(patient_vector)
            X_bmu_val_list.append(bmu_norm)
            y_val_list.append(i)
    
    X_sym_val = torch.FloatTensor(np.array(X_sym_val_list)).to(device)
    X_bmu_val = torch.FloatTensor(np.array(X_bmu_val_list)).to(device)
    y_val = torch.LongTensor(y_val_list)
    
    print(f"Training on {n_diseases} diseases")
    print(f"Validation on {len(y_val)} generated samples ({n_val_per_disease} per disease)")
    
    # Model
    model = TwoBranchNN(n_symptoms, n_diseases).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    history = {'val_acc': [], 'val_top5': [], 'val_top10': []}
    best_acc = 0
    best_state = None
    patience = 15
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_sym, batch_bmu, batch_y in train_loader:
            batch_sym = batch_sym.to(device)
            batch_bmu = batch_bmu.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(batch_sym, batch_bmu), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_sym_val, X_bmu_val)
            probs = torch.softmax(logits, dim=1)
            
            acc = (probs.argmax(1).cpu() == y_val).float().mean().item() * 100
            top5 = (probs.topk(5, 1).indices.cpu() == y_val.unsqueeze(1)).any(1).float().mean().item() * 100
            top10 = (probs.topk(10, 1).indices.cpu() == y_val.unsqueeze(1)).any(1).float().mean().item() * 100
        
        history['val_acc'].append(acc)
        history['val_top5'].append(top5)
        history['val_top10'].append(top10)
        
        # Early stopping
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Val={acc:.1f}%, Top5={top5:.1f}%, Top10={top10:.1f}%")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Відновлюємо найкращу модель
    if best_state:
        model.load_state_dict(best_state)
        print(f"Restored best model with Val={best_acc:.1f}%")
    
    return model, history


def main():
    print("=" * 60)
    print("ТРЕНУВАННЯ НА РЕАЛЬНИХ ДАНИХ")
    print("=" * 60)
    
    # Load database
    db_path = 'data/database_lowercase.json'
    db, disease_names, symptom_names = load_database(db_path)
    print(f"Diseases: {len(disease_names)}")
    print(f"Symptoms: {len(symptom_names)}")
    
    # Train SOM
    som = train_som(db, disease_names, symptom_names, grid_size=20, epochs=1000)
    
    # Save SOM у форматі сумісному з dr_case.som.SOMModel
    from dr_case.config import SOMConfig
    
    config = SOMConfig(
        grid_height=20,
        grid_width=20,
        input_dim=len(symptom_names),
        epochs=1000
    )
    
    # Конвертуємо disease_index у формати unit_to_diseases та disease_to_unit
    unit_to_diseases = {}
    disease_to_unit = {}
    
    for key, diseases in som._disease_index.items():
        parts = key.split('_')
        unit_tuple = (int(parts[0]), int(parts[1]))
        unit_to_diseases[unit_tuple] = diseases
        
        for disease in diseases:
            disease_to_unit[disease] = unit_tuple
    
    som_data = {
        "config": config,
        "som": som._som,
        "unit_to_diseases": unit_to_diseases,
        "disease_to_unit": disease_to_unit,
        "disease_names": som._disease_names,
        "is_trained": True,
        "qe": som._som.quantization_error(som._disease_vectors),
        "te": som._som.topographic_error(som._disease_vectors),
    }
    
    with open('models/som_real.pkl', 'wb') as f:
        pickle.dump(som_data, f)
    print("SOM saved to models/som_real.pkl")
    
    # Train NN
    model, history = train_nn(db, disease_names, symptom_names, som, epochs=200)
    
    # Save NN
    checkpoint = {
        'model_state': model.state_dict(),
        'model_config': {
            'n_symptoms': len(symptom_names),
            'n_diseases': len(disease_names),
            'som_dim': 2,
            'symptom_hidden': [256, 128],
            'bmu_hidden': [32, 16],
            'combined_hidden': [128],
            'dropout': 0.3
        },
        'disease_names': disease_names,
        'symptom_names': symptom_names,
        'som_shape': (20, 20),
    }
    torch.save(checkpoint, 'models/nn_real.pt')
    print("NN saved to models/nn_real.pt")
    
    # Verification
    print("\n" + "=" * 60)
    print("ПЕРЕВІРКА")
    print("=" * 60)
    
    for disease in ['covid-19', 'measles', 'influenza', 'pneumonia']:
        bmu = som.get_bmu_for_disease(disease)
        neighbors = som.get_diseases_at_bmu(bmu)[:5]
        print(f"{disease}: BMU={bmu}, neighbors={neighbors}")
    
    print("\nТест: fever + cough + headache")
    cluster = som.get_cluster_diseases(['fever', 'cough', 'headache'], radius=1)
    print(f"Diseases in cluster: {cluster[:10]}")
    
    print(f"\nFinal Val Accuracy: {history['val_acc'][-1]:.1f}%")
    print(f"Final Val Top-5: {history['val_top5'][-1]:.1f}%")
    print(f"Final Val Top-10: {history['val_top10'][-1]:.1f}%")


if __name__ == "__main__":
    main()
