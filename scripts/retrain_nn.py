"""
Dr.Case — Перенавчання Neural Network (Двогілкова архітектура)

Архітектура:
    symptom_vector (466) → [256 → 128] ─┐
                                         ├─→ concat → [128] → disease_scores (863)
    som_context (10) → [64 → 32] ───────┘

ВАЖЛИВО:
- symptom_vector: L2-нормалізується в моделі
- som_context: вже нормалізовано (softmax)

Запуск:
    python scripts/retrain_nn.py
"""

import sys
import os
from pathlib import Path

# Додаємо корінь проекту до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json

# Перевіряємо torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch не встановлено!")
    print("   Встановіть: pip install torch")
    sys.exit(1)

from dr_case.data_generation.two_branch_generator import (
    TwoBranchDataGenerator, 
    TwoBranchSamplerConfig
)
from dr_case.neural_network.two_branch_model import TwoBranchNN, TwoBranchDataset


def main():
    print("=" * 70)
    print("Dr.Case — ПЕРЕНАВЧАННЯ NEURAL NETWORK")
    print("Двогілкова архітектура: Симптоми + SOM Context")
    print("=" * 70)
    
    # Шляхи
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    som_path = project_root / "models" / "som_merged.pkl"
    output_model_path = project_root / "models" / "nn_two_branch.pt"
    output_history_path = project_root / "models" / "nn_two_branch_history.json"
    
    # Перевіряємо наявність файлів
    if not database_path.exists():
        print(f"❌ Файл не знайдено: {database_path}")
        return
    if not som_path.exists():
        print(f"❌ SOM модель не знайдена: {som_path}")
        print("   Спочатку запустіть: python scripts/retrain_som.py")
        return
    
    print(f"\n📁 База даних: {database_path}")
    print(f"📁 SOM модель: {som_path}")
    print(f"📁 Вихідна модель: {output_model_path}")
    
    # ========== КОНФІГУРАЦІЯ ==========
    
    # Генератор даних
    generator_config = TwoBranchSamplerConfig(
        samples_per_disease=100,      # 100 пацієнтів на хворобу
        min_symptoms=2,               # Мінімум 2 симптоми
        noise_probability=0.02,       # 2% шуму
        dropout_probability=0.15,     # 15% "забутих" симптомів
        som_k=10,                     # 10 top SOM юнітів
        som_alpha=0.95,
        som_tau=0.001,
        random_seed=42
    )
    
    # NN hyperparameters
    nn_config = {
        "symptom_hidden": [256, 128],
        "symptom_dropout": [0.3, 0.3],
        "som_hidden": [64, 32],
        "som_dropout": [0.2, 0.2],
        "combined_hidden": [128],
        "combined_dropout": 0.3,
        "use_batch_norm": True,
        "normalize_symptoms": True,  # L2-нормалізація
    }
    
    training_config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 100,
        "patience": 15,
        "min_delta": 1e-4,
    }
    
    print(f"\n⚙️ Конфігурація генератора:")
    print(f"   Samples per disease: {generator_config.samples_per_disease}")
    print(f"   SOM k: {generator_config.som_k}")
    print(f"   Noise prob: {generator_config.noise_probability}")
    print(f"   Dropout prob: {generator_config.dropout_probability}")
    
    print(f"\n⚙️ Конфігурація NN:")
    print(f"   Symptom branch: {nn_config['symptom_hidden']}")
    print(f"   SOM branch: {nn_config['som_hidden']}")
    print(f"   Combined: {nn_config['combined_hidden']}")
    print(f"   Normalize symptoms: {nn_config['normalize_symptoms']}")
    
    # ========== ГЕНЕРАЦІЯ ДАНИХ ==========
    
    print("\n" + "=" * 70)
    print("🔄 ГЕНЕРАЦІЯ ДАНИХ")
    print("=" * 70)
    
    generator = TwoBranchDataGenerator.from_files(
        str(database_path),
        str(som_path),
        generator_config
    )
    
    dims = generator.get_dimensions()
    print(f"\n📊 Розмірності:")
    print(f"   n_symptoms: {dims['n_symptoms']}")
    print(f"   n_diseases: {dims['n_diseases']}")
    print(f"   som_dim: {dims['som_dim']}")
    
    # Генеруємо train/val/test
    splits = generator.generate_train_val_test(
        samples_per_disease=generator_config.samples_per_disease,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        verbose=True
    )
    
    X_train_sym, X_train_som, y_train = splits["train"]
    X_val_sym, X_val_som, y_val = splits["val"]
    X_test_sym, X_test_som, y_test = splits["test"]
    
    print(f"\n📊 Розміри вибірок:")
    print(f"   Train: {len(y_train)} samples")
    print(f"   Val:   {len(y_val)} samples")
    print(f"   Test:  {len(y_test)} samples")
    
    # ========== СТВОРЕННЯ МОДЕЛІ ==========
    
    print("\n" + "=" * 70)
    print("🧠 СТВОРЕННЯ МОДЕЛІ")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = TwoBranchNN(
        n_symptoms=dims['n_symptoms'],
        n_diseases=dims['n_diseases'],
        som_dim=dims['som_dim'],
        **nn_config
    ).to(device)
    
    print(f"\n{model}")
    
    # ========== DATALOADERS ==========
    
    train_dataset = TwoBranchDataset(
        symptom_vectors=X_train_sym,
        som_contexts=X_train_som,
        disease_indices=y_train,
        augment=True,
        augment_prob=0.3,
        drop_prob=0.15,
        add_prob=0.03
    )
    
    val_dataset = TwoBranchDataset(
        symptom_vectors=X_val_sym,
        som_contexts=X_val_som,
        disease_indices=y_val,
        augment=False
    )
    
    test_dataset = TwoBranchDataset(
        symptom_vectors=X_test_sym,
        som_contexts=X_test_som,
        disease_indices=y_test,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # ========== TRAINING SETUP ==========
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # ========== TRAINING LOOP ==========
    
    print("\n" + "=" * 70)
    print("🔄 НАВЧАННЯ")
    print("=" * 70)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(training_config['epochs']):
        # === TRAIN ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for symptoms, som_context, targets in train_loader:
            symptoms = symptoms.to(device)
            som_context = som_context.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(symptoms, som_context)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * symptoms.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += symptoms.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for symptoms, som_context, targets in val_loader:
                symptoms = symptoms.to(device)
                som_context = som_context.to(device)
                targets = targets.to(device)
                
                outputs = model(symptoms, som_context)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * symptoms.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += symptoms.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # === HISTORY ===
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # === SCHEDULER ===
        scheduler.step(val_loss)
        
        # === EARLY STOPPING ===
        if val_loss < best_val_loss - training_config['min_delta']:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # === PRINT PROGRESS ===
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{training_config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
              f"LR: {lr:.2e}")
        
        # === CHECK EARLY STOPPING ===
        if epochs_without_improvement >= training_config['patience']:
            print(f"\n⏹ Early stopping at epoch {epoch + 1}")
            break
    
    # Завантажуємо найкращу модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # ========== EVALUATION ON TEST SET ==========
    
    print("\n" + "=" * 70)
    print("📊 ОЦІНКА НА ТЕСТОВІЙ ВИБІРЦІ")
    print("=" * 70)
    
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total = 0
    
    with torch.no_grad():
        for symptoms, som_context, targets in test_loader:
            symptoms = symptoms.to(device)
            som_context = som_context.to(device)
            targets = targets.to(device)
            
            outputs = model(symptoms, som_context)
            
            # Top-1
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(targets).sum().item()
            
            # Top-5
            _, pred_top5 = outputs.topk(5, dim=1)
            for i, target in enumerate(targets):
                if target in pred_top5[i]:
                    correct_top5 += 1
            
            # Top-10
            _, pred_top10 = outputs.topk(10, dim=1)
            for i, target in enumerate(targets):
                if target in pred_top10[i]:
                    correct_top10 += 1
            
            total += targets.size(0)
    
    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total
    acc_top10 = correct_top10 / total
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy@1:  {acc_top1:.2%}")
    print(f"   Accuracy@5:  {acc_top5:.2%}")
    print(f"   Accuracy@10: {acc_top10:.2%}")
    
    # ========== SAVE MODEL ==========
    
    print("\n" + "=" * 70)
    print("💾 ЗБЕРЕЖЕННЯ МОДЕЛІ")
    print("=" * 70)
    
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state': model.state_dict(),
        'model_config': model.config,
        'disease_names': generator.disease_names,
        'symptom_names': generator.symptom_names,
        'training_config': training_config,
        'generator_config': {
            'samples_per_disease': generator_config.samples_per_disease,
            'som_k': generator_config.som_k,
            'noise_probability': generator_config.noise_probability,
            'dropout_probability': generator_config.dropout_probability,
        },
        'metrics': {
            'test_accuracy_top1': acc_top1,
            'test_accuracy_top5': acc_top5,
            'test_accuracy_top10': acc_top10,
            'best_val_loss': best_val_loss,
        }
    }
    
    torch.save(checkpoint, output_model_path)
    print(f"   Model saved: {output_model_path}")
    
    # Save history
    history_data = {
        'config': {
            'nn': nn_config,
            'training': training_config,
            'generator': {
                'samples_per_disease': generator_config.samples_per_disease,
                'som_k': generator_config.som_k,
            }
        },
        'dimensions': dims,
        'metrics': checkpoint['metrics'],
        'history': history
    }
    
    with open(output_history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"   History saved: {output_history_path}")
    
    print("\n" + "=" * 70)
    print("✅ NEURAL NETWORK ПЕРЕНАВЧЕНО УСПІШНО!")
    print("=" * 70)
    
    print(f"\n📊 Підсумок:")
    print(f"   Архітектура: TwoBranchNN (Symptoms + SOM Context)")
    print(f"   Хвороб: {dims['n_diseases']}")
    print(f"   Симптомів: {dims['n_symptoms']}")
    print(f"   SOM context: {dims['som_dim']} юнітів")
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Test Accuracy@1: {acc_top1:.2%}")
    print(f"   Test Accuracy@5: {acc_top5:.2%}")
    print(f"   Test Accuracy@10: {acc_top10:.2%}")


if __name__ == "__main__":
    main()
