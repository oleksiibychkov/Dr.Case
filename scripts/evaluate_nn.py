"""
Dr.Case — Оцінка метрик Neural Network

Правильні метрики згідно CONFIG_PARAMETERS.md:
- Recall@1, Recall@5, Recall@10
- mAP (mean Average Precision)
- Hamming Loss (для multilabel)

Запуск:
    python scripts/evaluate_nn.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json

import torch
from torch.utils.data import DataLoader

from dr_case.data_generation.two_branch_generator import (
    TwoBranchDataGenerator, 
    TwoBranchSamplerConfig
)
from dr_case.neural_network.two_branch_model import TwoBranchNN, TwoBranchDataset


def recall_at_k(predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
    """
    Recall@k для single-label classification.
    
    Recall@k = (кількість samples де правильний клас в топ-k) / (всього samples)
    
    Для single-label це еквівалентно "hit rate" або "top-k accuracy".
    """
    n_samples = len(targets)
    hits = 0
    
    for i in range(n_samples):
        top_k_indices = np.argsort(predictions[i])[-k:][::-1]
        if targets[i] in top_k_indices:
            hits += 1
    
    return hits / n_samples


def average_precision(predictions: np.ndarray, target: int) -> float:
    """
    Average Precision для одного sample (single-label).
    
    AP = 1 / rank(correct_class)
    
    Чим вище правильний клас у ранжуванні, тим вище AP.
    """
    sorted_indices = np.argsort(predictions)[::-1]
    rank = np.where(sorted_indices == target)[0][0] + 1  # 1-based rank
    return 1.0 / rank


def mean_average_precision(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Average Precision (mAP).
    
    mAP = mean(AP для кожного sample)
    """
    n_samples = len(targets)
    ap_sum = 0.0
    
    for i in range(n_samples):
        ap_sum += average_precision(predictions[i], targets[i])
    
    return ap_sum / n_samples


def mean_reciprocal_rank(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Reciprocal Rank (MRR).
    
    MRR = mean(1 / rank(correct_class))
    
    Для single-label MRR = mAP.
    """
    return mean_average_precision(predictions, targets)


def rank_distribution(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Розподіл рангів правильних класів.
    """
    ranks = []
    
    for i in range(len(targets)):
        sorted_indices = np.argsort(predictions[i])[::-1]
        rank = np.where(sorted_indices == targets[i])[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    return {
        "min": int(ranks.min()),
        "max": int(ranks.max()),
        "mean": float(ranks.mean()),
        "median": float(np.median(ranks)),
        "std": float(ranks.std()),
        "rank_1": int((ranks == 1).sum()),
        "rank_1_5": int((ranks <= 5).sum()),
        "rank_1_10": int((ranks <= 10).sum()),
        "rank_1_20": int((ranks <= 20).sum()),
    }


def main():
    print("=" * 70)
    print("Dr.Case — ОЦІНКА МЕТРИК NEURAL NETWORK")
    print("=" * 70)
    
    # Шляхи
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    som_path = project_root / "models" / "som_merged.pkl"
    model_path = project_root / "models" / "nn_two_branch.pt"
    output_path = project_root / "models" / "nn_evaluation_metrics.json"
    
    # Перевірка файлів
    for path, name in [(database_path, "Database"), (som_path, "SOM"), (model_path, "NN Model")]:
        if not path.exists():
            print(f"❌ {name} не знайдено: {path}")
            return
    
    print(f"\n📁 Model: {model_path}")
    
    # ========== ЗАВАНТАЖЕННЯ МОДЕЛІ ==========
    
    print("\n🔄 Завантаження моделі...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = TwoBranchNN(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"   Device: {device}")
    print(f"   Parameters: {model.count_parameters():,}")
    
    # ========== ГЕНЕРАЦІЯ ТЕСТОВИХ ДАНИХ ==========
    
    print("\n🔄 Генерація тестових даних...")
    
    generator_config = TwoBranchSamplerConfig(
        samples_per_disease=50,  # Менше для швидкої оцінки
        min_symptoms=2,
        noise_probability=0.02,
        dropout_probability=0.15,
        som_k=10,
        random_seed=123  # Інший seed для тесту
    )
    
    generator = TwoBranchDataGenerator.from_files(
        str(database_path),
        str(som_path),
        generator_config
    )
    
    # Генеруємо тестову вибірку
    X_sym, X_som, y, _ = generator.generate(
        samples_per_disease=50,
        verbose=True
    )
    
    print(f"   Test samples: {len(y)}")
    
    # ========== ОТРИМАННЯ PREDICTIONS ==========
    
    print("\n🔄 Отримання predictions...")
    
    dataset = TwoBranchDataset(
        symptom_vectors=X_sym,
        som_contexts=X_som,
        disease_indices=y,
        augment=False
    )
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for symptoms, som_context, targets in loader:
            symptoms = symptoms.to(device)
            som_context = som_context.to(device)
            
            outputs = model(symptoms, som_context)
            
            # Softmax для ймовірностей
            probs = torch.softmax(outputs, dim=1)
            
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.concatenate(all_targets)
    
    print(f"   Predictions shape: {predictions.shape}")
    
    # ========== ОБЧИСЛЕННЯ МЕТРИК ==========
    
    print("\n" + "=" * 70)
    print("📊 МЕТРИКИ")
    print("=" * 70)
    
    # Recall@k
    recall_1 = recall_at_k(predictions, targets, k=1)
    recall_5 = recall_at_k(predictions, targets, k=5)
    recall_10 = recall_at_k(predictions, targets, k=10)
    recall_20 = recall_at_k(predictions, targets, k=20)
    
    print(f"\n📈 Recall@k:")
    print(f"   Recall@1:  {recall_1:.2%}  (ціль прототипу: 50%, production: 70%)")
    print(f"   Recall@5:  {recall_5:.2%}  (ціль прототипу: 85%, production: 95%)")
    print(f"   Recall@10: {recall_10:.2%}  (ціль прототипу: 92%, production: 99%)")
    print(f"   Recall@20: {recall_20:.2%}")
    
    # mAP
    map_score = mean_average_precision(predictions, targets)
    mrr_score = mean_reciprocal_rank(predictions, targets)
    
    print(f"\n📈 Ranking Quality:")
    print(f"   mAP (mean Average Precision): {map_score:.4f}  (ціль прототипу: 0.60, production: 0.80)")
    print(f"   MRR (Mean Reciprocal Rank):   {mrr_score:.4f}")
    
    # Rank distribution
    rank_dist = rank_distribution(predictions, targets)
    
    print(f"\n📈 Rank Distribution:")
    print(f"   Mean rank: {rank_dist['mean']:.1f}")
    print(f"   Median rank: {rank_dist['median']:.1f}")
    print(f"   Min/Max rank: {rank_dist['min']}/{rank_dist['max']}")
    print(f"   Rank = 1: {rank_dist['rank_1']} ({rank_dist['rank_1']/len(targets):.1%})")
    print(f"   Rank ≤ 5: {rank_dist['rank_1_5']} ({rank_dist['rank_1_5']/len(targets):.1%})")
    print(f"   Rank ≤ 10: {rank_dist['rank_1_10']} ({rank_dist['rank_1_10']/len(targets):.1%})")
    
    # ========== ПОРІВНЯННЯ З ЦІЛЯМИ ==========
    
    print("\n" + "=" * 70)
    print("🎯 ПОРІВНЯННЯ З ЦІЛЯМИ")
    print("=" * 70)
    
    targets_prototype = {
        "Recall@1": 0.50,
        "Recall@5": 0.85,
        "Recall@10": 0.92,
        "mAP": 0.60,
    }
    
    targets_production = {
        "Recall@1": 0.70,
        "Recall@5": 0.95,
        "Recall@10": 0.99,
        "mAP": 0.80,
    }
    
    actual = {
        "Recall@1": recall_1,
        "Recall@5": recall_5,
        "Recall@10": recall_10,
        "mAP": map_score,
    }
    
    print(f"\n{'Metric':<12} {'Actual':>10} {'Prototype':>12} {'Production':>12} {'Status'}")
    print("-" * 60)
    
    for metric in ["Recall@1", "Recall@5", "Recall@10", "mAP"]:
        val = actual[metric]
        proto = targets_prototype[metric]
        prod = targets_production[metric]
        
        if val >= prod:
            status = "✅ Production"
        elif val >= proto:
            status = "✅ Prototype"
        else:
            status = "❌ Below"
        
        print(f"{metric:<12} {val:>10.2%} {proto:>12.0%} {prod:>12.0%} {status}")
    
    # ========== ЗБЕРЕЖЕННЯ ==========
    
    metrics_data = {
        "model": str(model_path),
        "test_samples": len(targets),
        "recall": {
            "recall_1": recall_1,
            "recall_5": recall_5,
            "recall_10": recall_10,
            "recall_20": recall_20,
        },
        "ranking": {
            "mAP": map_score,
            "MRR": mrr_score,
        },
        "rank_distribution": rank_dist,
        "targets": {
            "prototype": targets_prototype,
            "production": targets_production,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\n💾 Metrics saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ ОЦІНКА ЗАВЕРШЕНА!")
    print("=" * 70)


if __name__ == "__main__":
    main()
