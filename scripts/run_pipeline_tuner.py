#!/usr/bin/env python3
"""
Dr.Case — Запуск Full Pipeline Tuner

Автоматична оптимізація параметрів всього pipeline:
- SOM grid size, learning rate, sigma
- Candidate Selector alpha, k
- Neural Network architecture, dropout, learning rate

Запуск:
    python scripts/run_pipeline_tuner.py
    python scripts/run_pipeline_tuner.py --only-nn          # Тільки NN
    python scripts/run_pipeline_tuner.py --only-som         # Тільки SOM
    python scripts/run_pipeline_tuner.py --quick            # Швидкий режим

Час виконання:
    - Повний: ~1-2 години
    - Quick: ~20-30 хвилин
    - Тільки NN: ~30-40 хвилин
"""

import sys
import argparse
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Tuner')
    parser.add_argument('--only-som', action='store_true', help='Оптимізувати тільки SOM')
    parser.add_argument('--only-nn', action='store_true', help='Оптимізувати тільки NN')
    parser.add_argument('--only-candidate', action='store_true', help='Оптимізувати тільки Candidate Selector')
    parser.add_argument('--quick', action='store_true', help='Швидкий режим (менше варіантів)')
    parser.add_argument('--restore-backup', action='store_true', help='Відновити моделі з backup')
    
    args = parser.parse_args()
    
    # Шляхи
    database_path = project_root / "data" / "unified_disease_symptom_merged.json"
    som_path = project_root / "models" / "som_merged.pkl"
    nn_path = project_root / "models" / "nn_two_branch.pt"
    output_dir = project_root / "models" / "tuning_results"
    
    # Відновлення з backup
    if args.restore_backup:
        import shutil
        backup_som = som_path.with_suffix('.pkl.backup')
        backup_nn = nn_path.with_suffix('.pt.backup')
        
        if backup_som.exists():
            shutil.copy(backup_som, som_path)
            print(f"✅ Відновлено SOM з backup")
        else:
            print(f"⚠️ Backup SOM не знайдено: {backup_som}")
        
        if backup_nn.exists():
            shutil.copy(backup_nn, nn_path)
            print(f"✅ Відновлено NN з backup")
        else:
            print(f"⚠️ Backup NN не знайдено: {backup_nn}")
        
        return
    
    # Перевірка файлів
    if not database_path.exists():
        print(f"❌ База даних не знайдена: {database_path}")
        return
    
    print("=" * 70)
    print("Dr.Case — FULL PIPELINE TUNER")
    print("=" * 70)
    
    # Визначаємо що оптимізувати
    tune_som = not args.only_nn and not args.only_candidate
    tune_candidate = not args.only_som and not args.only_nn
    tune_nn = not args.only_som and not args.only_candidate
    
    if args.only_som:
        tune_som, tune_candidate, tune_nn = True, False, False
    elif args.only_nn:
        tune_som, tune_candidate, tune_nn = False, False, True
    elif args.only_candidate:
        tune_som, tune_candidate, tune_nn = False, True, False
    
    print(f"\n📋 Конфігурація:")
    print(f"   Tune SOM: {tune_som}")
    print(f"   Tune Candidate: {tune_candidate}")
    print(f"   Tune NN: {tune_nn}")
    print(f"   Quick mode: {args.quick}")
    
    # Імпорти
    from dr_case.optimization.full_pipeline_tuner import FullPipelineTuner, TuningConfig
    
    # Конфігурація
    if args.quick:
        config = TuningConfig(
            # Менше варіантів для швидкого режиму
            som_grid_sizes=[(15, 15), (20, 20)],
            som_learning_rates=[0.5],
            som_sigma_init=[5.0],
            candidate_alphas=[0.90, 0.95],
            candidate_k=[8, 10],
            nn_hidden_dims=[[256, 128]],
            nn_dropout=[0.3],
            nn_learning_rates=[1e-3],
            max_epochs=50,
            patience=5,
            n_validation_samples=200,
        )
    else:
        config = TuningConfig(
            som_grid_sizes=[(12, 12), (15, 15), (18, 18), (20, 20)],
            som_learning_rates=[0.3, 0.5, 0.7],
            som_sigma_init=[3.0, 5.0, 7.0],
            candidate_alphas=[0.85, 0.90, 0.95],
            candidate_k=[6, 8, 10, 12],
            nn_hidden_dims=[[256, 128], [512, 256], [256, 128, 64]],
            nn_dropout=[0.2, 0.3, 0.4],
            nn_learning_rates=[1e-3, 5e-4],
            max_epochs=100,
            patience=10,
            n_validation_samples=500,
        )
    
    # Оцінка часу
    n_som_trials = (len(config.som_grid_sizes) * 
                   len(config.som_learning_rates) * 
                   len(config.som_sigma_init)) if tune_som else 0
    n_candidate_trials = (len(config.candidate_alphas) * 
                         len(config.candidate_k)) if tune_candidate else 0
    n_nn_trials = (len(config.nn_hidden_dims) * 
                  len(config.nn_dropout) * 
                  len(config.nn_learning_rates)) if tune_nn else 0
    
    print(f"\n📊 Очікувані trials:")
    if tune_som:
        print(f"   SOM: {n_som_trials} trials (~{n_som_trials * 2} хв)")
    if tune_candidate:
        print(f"   Candidate: {n_candidate_trials} trials (~{n_candidate_trials} хв)")
    if tune_nn:
        print(f"   NN: {n_nn_trials} trials (~{n_nn_trials * 5} хв)")
    
    total_minutes = (n_som_trials * 2 + n_candidate_trials + n_nn_trials * 5)
    print(f"\n⏱️ Орієнтовний час: {total_minutes} хвилин ({total_minutes/60:.1f} годин)")
    
    # Підтвердження
    response = input("\n▶ Продовжити? (y/n): ").strip().lower()
    if response != 'y':
        print("Скасовано.")
        return
    
    # Створюємо tuner
    tuner = FullPipelineTuner(config)
    
    # Запускаємо
    start_time = time.time()
    
    try:
        report = tuner.tune(
            database_path=str(database_path),
            output_dir=str(output_dir),
            som_path=str(som_path) if som_path.exists() else None,
            nn_path=str(nn_path) if nn_path.exists() else None,
            strategy="iterative",
            tune_som=tune_som,
            tune_candidate=tune_candidate,
            tune_nn=tune_nn,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        # Результати
        print("\n" + "=" * 70)
        print("🎉 TUNING ЗАВЕРШЕНО!")
        print("=" * 70)
        
        print(f"\n📊 Результати:")
        print(f"   Початкова якість: {report.initial_score:.4f}")
        print(f"   Фінальна якість:  {report.final_score:.4f}")
        print(f"   Покращення:       {report.total_improvement:+.4f}")
        print(f"   Час:              {elapsed/60:.1f} хвилин")
        
        if report.som_result:
            print(f"\n   SOM:")
            print(f"      Best params: {report.som_result.best_params}")
            print(f"      Best score: {report.som_result.best_score:.4f}")
        
        if report.candidate_result:
            print(f"\n   Candidate:")
            print(f"      Best params: {report.candidate_result.best_params}")
            print(f"      Best score: {report.candidate_result.best_score:.4f}")
        
        if report.nn_result:
            print(f"\n   NN:")
            print(f"      Best params: {report.nn_result.best_params}")
            print(f"      Best score: {report.nn_result.best_score:.4f}")
        
        # Збереження звіту
        report_path = output_dir / "tuning_report.json"
        report.save(str(report_path))
        print(f"\n💾 Звіт збережено: {report_path}")
        
        # Копіювання найкращих моделей
        print(f"\n💡 Найкращі моделі збережено в: {output_dir}")
        print(f"   - som_tuned.pkl")
        print(f"   - nn_tuned.pt")
        
        print(f"\n📋 Наступні кроки:")
        print(f"   1. Перевірте результати: python scripts/test_validation.py")
        print(f"   2. Якщо результати кращі — скопіюйте моделі:")
        print(f"      copy {output_dir}\\som_tuned.pkl models\\som_merged.pkl")
        print(f"      copy {output_dir}\\nn_tuned.pt models\\nn_two_branch.pt")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
