"""
Перетренування SOM і NN після виправлення бази до lowercase
"""

import json
import pickle

from dr_case.som import SOMModel
from dr_case.encoding import DiseaseEncoder
from dr_case.multilabel_nn import NNTrainer_BMU


def main():
    print("=" * 60)
    print("ПЕРЕТРЕНУВАННЯ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Завантаження бази
    print("\n[1] Завантаження бази...")
    with open('data/database_lowercase.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
    print(f"    Діагнозів: {len(db)}")
    
    # Encoder
    print("\n[2] Створення encoder...")
    encoder = DiseaseEncoder.from_database(db)
    vectors = encoder.encode_all_diseases()
    print(f"    Vectors shape: {vectors.shape}")
    
    # SOM
    print("\n[3] Навчання SOM 20x20...")
    som = SOMModel(20, 20, vectors.shape[1])
    som.train(vectors, epochs=500)
    som.build_disease_index(encoder.disease_names, vectors)
    
    qe = som.compute_quantization_error(vectors)
    te = som.compute_topographic_error(vectors)
    fill = som.get_fill_ratio()
    print(f"    QE: {qe:.4f}")
    print(f"    TE: {te:.4f}")
    print(f"    Fill: {fill:.1%}")
    
    with open('models/som_lowercase.pkl', 'wb') as f:
        pickle.dump(som, f)
    print("    Збережено: models/som_lowercase.pkl")
    
    # NN
    print("\n[4] Навчання NN...")
    trainer = NNTrainer_BMU(db, som, encoder)
    model, history = trainer.train(epochs=50, batch_size=64)
    trainer.save_model(model, 'models/nn_lowercase.pt', history)
    print("    Збережено: models/nn_lowercase.pt")
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
