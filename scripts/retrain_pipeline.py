#!/usr/bin/env python3
"""
Dr.Case — Повне перенавчання Pipeline (SOM 15x15 + NN)

Етапи:
1. Перенавчання SOM з оптимальним grid 15x15 (замість 30x30)
2. Перенавчання NN з новим SOM
3. Валідація результатів

Очікуваний час: ~25-30 хвилин

Запуск:
    python scripts/retrain_pipeline.py
"""

import sys
import os
from pathlib import Path
import time
import json
import pickle

# Додаємо корінь проекту до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

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


def retrain_som(database_path: Path, output_path: Path, grid_size: int = 15):
    """
    Перенавчання SOM з оптимальним grid size.
    
    Формула Vesanto: grid ≈ 5 × √N
    Для 844 хвороб: 5 × √844 ≈ 145 юнітів → 12×12
    Беремо 15×15 = 225 для запасу
    """
    from dr_case.config import SOMConfig
    from dr_case.som import SOMTrainer
    
    print("\n" + "=" * 70)
    print(f"🔄 ЕТАП 1: ПЕРЕНАВЧАННЯ SOM ({grid_size}x{grid_size})")
    print("=" * 70)
    
    start_time = time.time()
    
    # Конфігурація SOM (оптимізована)
    config = SOMConfig(
        grid_height=grid_size,
        grid_width=grid_size,
        input_dim=460,               # Кількість симптомів
        epochs=500,                  # Достатньо для збіжності
        learning_rate_init=0.5,
        learning_rate_final=0.01,
        sigma_init=grid_size / 2,    # Початковий радіус = grid/2
        sigma_final=0.5,
    )
    
    print(f"\n⚙️ Конфігурація SOM:")
    print(f"   Grid: {config.grid_height}x{config.grid_width} = {config.grid_height * config.grid_width} юнітів")
    print(f"   Epochs: {config.epochs}")
    print(f"   Learning rate: {config.learning_rate_init} → {config.learning_rate_final}")
    print(f"   Sigma: {config.sigma_init} → {config.sigma_final}")
    
    # Створюємо тренер
    trainer = SOMTrainer(config)
    
    # Навчаємо
    print("\n🔄 Навчання SOM...")
    print("-" * 70)
    
    som_model, metrics = trainer.train_from_database(
        str(database_path),
        epochs=config.epochs,
        verbose=True
    )
    
    print("-" * 70)
    print(f"\n📊 Результати SOM:")
    print(f"   QE (Quantization Error): {metrics['qe']:.4f}")
    print(f"   TE (Topographic Error): {metrics['te']:.4f}")
    print(f"   Fill ratio: {metrics['fill_ratio']:.2%}")
    print(f"   Filled units: {metrics['filled_units']}/{metrics['total_units']}")
    
    # Оцінюємо Candidate Recall
    print("\n📈 Оцінка Candidate Recall...")
    recall_metrics = trainer.evaluate_candidate_recall(alpha=0.95, k=10, tau=0.001)
    print(f"   Recall: {recall_metrics['recall']:.4f} ({recall_metrics['hits']}/{recall_metrics['total']})")
    print(f"   Avg candidates: {recall_metrics['avg_candidates']:.1f}")
    
    # Зберігаємо модель
    print(f"\n💾 Збереження моделі: {output_path}")
    trainer.save_model(str(output_path))
    
    elapsed = time.time() - start_time
    print(f"\n✅ SOM перенавчено за {elapsed:.1f}с")
    
    return metrics, recall_metrics


def retrain_nn(database_path: Path, som_path: Path, output_path: Path):
    """
    Перенавчання Neural Network з новим SOM.
    """
    from dr_case.data_generation.two_branch_generator import (
        TwoBranchDataGenerator, 
        TwoBranchSamplerConfig
    )
    from dr_case.neural_network.two_branch_model import TwoBranchNN, TwoBranchDataset
    
    print("\n" + "=" * 70)
    print("🧠 ЕТАП 2: ПЕРЕНАВЧАННЯ NEURAL NETWORK")
    print("=" * 70)
    
    start_time = time.time()
    
    # Конфігурація
    generator_config = TwoBranchSamplerConfig(
        samples_per_disease=100,
        min_symptoms=2,
        noise_probability=0.02,
        dropout_probability=0.15,
        som_k=10,
        som_alpha=0.95,
        som_tau=0.001,
        random_seed=42
    )
    
    nn_config = {
        "symptom_hidden": [256, 128],
        "symptom_dropout": [0.3, 0.3],
        "som_hidden": [64, 32],
        "som_dropout": [0.2, 0.2],
        "combined_hidden": [128],
        "combined_dropout": 0.3,
        "use_batch_norm": True,
        "normalize_symptoms": True,
    }
    
    training_config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 100,
        "patience": 15,
        "min_delta": 1e-4,
    }
    
    print(f"\n⚙️ Конфігурація NN:")
    print(f"   Symptom branch: {nn_config['symptom_hidden']}")
    print(f"   SOM branch: {nn_config['som_hidden']}")
    print(f"   Combined: {nn_config['combined_hidden']}")
    print(f"   Samples per disease: {generator_config.samples_per_disease}")
    
    # ========== ГЕНЕРАЦІЯ ДАНИХ ==========
    
    print("\n🔄 Генерація даних з новим SOM...")
    
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
    
    print(f"\n📊 Розміри даних:")
    print(f"   Train: {len(y_train)}")
    print(f"   Val: {len(y_val)}")
    print(f"   Test: {len(y_test)}")
    
    # ========== СТВОРЕННЯ МОДЕЛІ ==========
    
    n_symptoms = dims['n_symptoms']
    n_diseases = dims['n_diseases']
    som_dim = dims['som_dim']
    
    model = TwoBranchNN(
        n_symptoms=n_symptoms,
        som_dim=som_dim,
        n_diseases=n_diseases,
        symptom_hidden=nn_config['symptom_hidden'],
        symptom_dropout=nn_config['symptom_dropout'],
        som_hidden=nn_config['som_hidden'],
        som_dropout=nn_config['som_dropout'],
        combined_hidden=nn_config['combined_hidden'],
        combined_dropout=nn_config['combined_dropout'],
        use_batch_norm=nn_config['use_batch_norm'],
        normalize_symptoms=nn_config['normalize_symptoms'],
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Модель створено:")
    print(f"   Parameters: {total_params:,}")
    
    # ========== ДАТАСЕТИ ==========
    
    train_dataset = TwoBranchDataset(X_train_sym, X_train_som, y_train)
    val_dataset = TwoBranchDataset(X_val_sym, X_val_som, y_val)
    test_dataset = TwoBranchDataset(X_test_sym, X_test_som, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config['batch_size']
    )
    
    # ========== ТРЕНУВАННЯ ==========
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print("\n🔄 Тренування...")
    print("-" * 70)
    
    for epoch in range(training_config['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for symptoms, som_ctx, targets in train_loader:
            symptoms = symptoms.to(device)
            som_ctx = som_ctx.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(symptoms, som_ctx)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * symptoms.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for symptoms, som_ctx, targets in val_loader:
                symptoms = symptoms.to(device)
                som_ctx = som_ctx.to(device)
                targets = targets.to(device)
                
                outputs = model(symptoms, som_ctx)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * symptoms.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - training_config['min_delta']:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")
        
        if patience_counter >= training_config['patience']:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    print("-" * 70)
    
    # Завантажуємо найкращу модель
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # ========== ТЕСТУВАННЯ ==========
    
    print("\n📊 Тестування на test set...")
    
    model.eval()
    test_correct_1 = 0
    test_correct_5 = 0
    test_correct_10 = 0
    test_total = 0
    
    with torch.no_grad():
        for symptoms, som_ctx, targets in test_loader:
            symptoms = symptoms.to(device)
            som_ctx = som_ctx.to(device)
            targets = targets.to(device)
            
            outputs = model(symptoms, som_ctx)
            
            # Top-1
            _, top1 = outputs.max(1)
            test_correct_1 += top1.eq(targets).sum().item()
            
            # Top-5
            _, top5 = outputs.topk(5, dim=1)
            test_correct_5 += sum(targets[i].item() in top5[i].tolist() 
                                  for i in range(targets.size(0)))
            
            # Top-10
            _, top10 = outputs.topk(10, dim=1)
            test_correct_10 += sum(targets[i].item() in top10[i].tolist() 
                                   for i in range(targets.size(0)))
            
            test_total += targets.size(0)
    
    test_acc_1 = test_correct_1 / test_total
    test_acc_5 = test_correct_5 / test_total
    test_acc_10 = test_correct_10 / test_total
    
    print(f"\n📊 Результати NN:")
    print(f"   Test Accuracy@1:  {test_acc_1:.2%}")
    print(f"   Test Accuracy@5:  {test_acc_5:.2%}")
    print(f"   Test Accuracy@10: {test_acc_10:.2%}")
    
    # ========== ЗБЕРЕЖЕННЯ ==========
    
    print(f"\n💾 Збереження моделі: {output_path}")
    
    checkpoint = {
        'model_state': best_model_state or model.state_dict(),
        'model_config': {
            'n_symptoms': n_symptoms,
            'n_diseases': n_diseases,
            'som_dim': som_dim,
            **nn_config
        },
        'disease_names': generator.disease_names,
        'symptom_names': generator.symptom_names,
        'training_config': training_config,
        'generator_config': {
            'samples_per_disease': generator_config.samples_per_disease,
            'min_symptoms': generator_config.min_symptoms,
            'noise_probability': generator_config.noise_probability,
            'dropout_probability': generator_config.dropout_probability,
            'som_k': generator_config.som_k,
        },
        'metrics': {
            'test_accuracy_1': test_acc_1,
            'test_accuracy_5': test_acc_5,
            'test_accuracy_10': test_acc_10,
            'best_val_loss': best_val_loss,
        }
    }
    
    torch.save(checkpoint, output_path)
    
    elapsed = time.time() - start_time
    print(f"\n✅ NN перенавчено за {elapsed:.1f}с")
    
    return {
        'test_acc_1': test_acc_1,
        'test_acc_5': test_acc_5,
        'test_acc_10': test_acc_10,
    }


def validate_pipeline(database_path: Path, som_path: Path, nn_path: Path):
    """
    Швидка валідація pipeline після перенавчання.
    """
    print("\n" + "=" * 70)
    print("✅ ЕТАП 3: ВАЛІДАЦІЯ PIPELINE")
    print("=" * 70)
    
    import pickle
    
    # Завантажуємо моделі
    with open(som_path, 'rb') as f:
        som_data = pickle.load(f)
    
    nn_checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
    
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    # Статистика SOM
    som = som_data['som']
    unit_to_diseases = som_data['unit_to_diseases']
    h, w = som._weights.shape[:2]
    
    print(f"\n📊 SOM:")
    print(f"   Grid: {h}x{w} = {h*w} юнітів")
    print(f"   Активних юнітів: {len(unit_to_diseases)}")
    print(f"   Fill rate: {len(unit_to_diseases) / (h*w):.1%}")
    print(f"   QE: {som_data.get('qe', 'N/A')}")
    print(f"   TE: {som_data.get('te', 'N/A')}")
    
    # Статистика NN
    metrics = nn_checkpoint.get('metrics', {})
    print(f"\n📊 Neural Network:")
    print(f"   Test Accuracy@1:  {metrics.get('test_accuracy_1', 0):.2%}")
    print(f"   Test Accuracy@5:  {metrics.get('test_accuracy_5', 0):.2%}")
    print(f"   Test Accuracy@10: {metrics.get('test_accuracy_10', 0):.2%}")
    
    # Тест на 20 випадках через DiagnosisCycleController
    print("\n🧪 E2E тест (20 випадків)...")
    
    try:
        from dr_case.diagnosis_cycle import DiagnosisCycleController
        
        controller = DiagnosisCycleController.from_models(
            database_path=str(database_path),
            som_path=str(som_path),
            nn_path=str(nn_path),
            language="uk"
        )
        
        np.random.seed(42)
        diseases = list(database.keys())
        test_diseases = np.random.choice(diseases, size=20, replace=False)
        
        hits_1, hits_5, hits_10 = 0, 0, 0
        
        for disease in test_diseases:
            symptoms = database[disease].get('symptoms', [])
            if not symptoms:
                continue
            
            initial = symptoms[:5]
            
            try:
                controller.start_session(initial)
                hypotheses = controller.get_top_hypotheses(10)
                top_names = [h[0] for h in hypotheses]
                
                if disease in top_names[:1]:
                    hits_1 += 1
                if disease in top_names[:5]:
                    hits_5 += 1
                if disease in top_names[:10]:
                    hits_10 += 1
            except:
                pass
        
        print(f"\n📊 E2E Results:")
        print(f"   Accuracy@1:  {hits_1}/20 = {hits_1/20:.1%}")
        print(f"   Accuracy@5:  {hits_5}/20 = {hits_5/20:.1%}")
        print(f"   Accuracy@10: {hits_10}/20 = {hits_10/20:.1%}")
        
    except Exception as e:
        print(f"   ⚠️ E2E тест пропущено: {e}")


def main():
    print("=" * 70)
    print("Dr.Case — ПОВНЕ ПЕРЕНАВЧАННЯ PIPELINE")
    print("SOM 15x15 + Neural Network")
    print("=" * 70)
    
    total_start = time.time()
    
    # Шляхи
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    som_path = project_root / "models" / "som_merged.pkl"
    nn_path = project_root / "models" / "nn_two_branch.pt"
    
    # Перевіряємо наявність даних
    if not database_path.exists():
        print(f"❌ Файл не знайдено: {database_path}")
        return
    
    print(f"\n📁 База даних: {database_path}")
    print(f"📁 SOM модель: {som_path}")
    print(f"📁 NN модель: {nn_path}")
    
    # Бекап старих моделей
    import shutil
    if som_path.exists():
        backup_som = som_path.with_suffix('.pkl.backup')
        shutil.copy(som_path, backup_som)
        print(f"\n💾 Backup SOM: {backup_som}")
    
    if nn_path.exists():
        backup_nn = nn_path.with_suffix('.pt.backup')
        shutil.copy(nn_path, backup_nn)
        print(f"💾 Backup NN: {backup_nn}")
    
    # ЕТАП 1: Перенавчання SOM
    som_metrics, recall_metrics = retrain_som(
        database_path, 
        som_path, 
        grid_size=15  # Оптимальний розмір
    )
    
    # ЕТАП 2: Перенавчання NN
    nn_metrics = retrain_nn(database_path, som_path, nn_path)
    
    # ЕТАП 3: Валідація
    validate_pipeline(database_path, som_path, nn_path)
    
    # Підсумок
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("🎉 ПЕРЕНАВЧАННЯ ЗАВЕРШЕНО!")
    print("=" * 70)
    
    print(f"\n📊 Підсумок:")
    print(f"   Загальний час: {total_elapsed/60:.1f} хвилин")
    print(f"\n   SOM (15x15):")
    print(f"      QE: {som_metrics['qe']:.4f}")
    print(f"      Fill: {som_metrics['fill_ratio']:.1%}")
    print(f"      Candidate Recall: {recall_metrics['recall']:.1%}")
    print(f"\n   Neural Network:")
    print(f"      Accuracy@1:  {nn_metrics['test_acc_1']:.1%}")
    print(f"      Accuracy@5:  {nn_metrics['test_acc_5']:.1%}")
    print(f"      Accuracy@10: {nn_metrics['test_acc_10']:.1%}")
    
    print(f"\n💡 Наступні кроки:")
    print(f"   1. Запустіть: python scripts/test_validation.py")
    print(f"   2. Якщо результати задовільні — продовжуйте з етапами 16-18")
    print(f"   3. Якщо потрібно відновити старі моделі:")
    print(f"      - cp models/som_merged.pkl.backup models/som_merged.pkl")
    print(f"      - cp models/nn_two_branch.pt.backup models/nn_two_branch.pt")


if __name__ == "__main__":
    main()
