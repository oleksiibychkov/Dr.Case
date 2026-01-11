"""
Dr.Case — Двогілкова архітектура Neural Network

Архітектура згідно PROJECT_ARCHITECTURE.md:
- Гілка 1: symptom_vector (466) → [256, 128] 
- Гілка 2: som_membership (k юнітів) → [64, 32]
- Об'єднання → [128] → disease_scores (863)

ВАЖЛИВО: Нормалізація входів!
- symptom_vector: L2-нормалізація
- som_membership: вже нормалізовано (softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class TwoBranchNN(nn.Module):
    """
    Двогілкова нейронна мережа для діагностики.
    
    Вхід 1: symptom_vector (D симптомів) - бінарний вектор симптомів
    Вхід 2: som_context (k юнітів) - membership до top-k SOM юнітів
    
    Архітектура:
        symptom_vector → [256 → 128] ─┐
                                       ├─→ concat → [128] → disease_scores
        som_context → [64 → 32] ──────┘
    
    Приклад використання:
        model = TwoBranchNN(
            n_symptoms=466,
            n_diseases=863,
            som_dim=10  # k=10 юнітів
        )
        
        symptoms = torch.randn(batch_size, 466)
        som_context = torch.randn(batch_size, 10)
        
        scores = model(symptoms, som_context)
    """
    
    def __init__(
        self,
        n_symptoms: int = 466,
        n_diseases: int = 863,
        som_dim: int = 10,
        # Гілка симптомів
        symptom_hidden: List[int] = None,
        symptom_dropout: List[float] = None,
        # Гілка SOM
        som_hidden: List[int] = None,
        som_dropout: List[float] = None,
        # Об'єднана частина
        combined_hidden: List[int] = None,
        combined_dropout: float = 0.3,
        # Загальні налаштування
        use_batch_norm: bool = True,
        normalize_symptoms: bool = True,  # L2-нормалізація симптомів
    ):
        """
        Args:
            n_symptoms: Кількість симптомів (вхід гілки 1)
            n_diseases: Кількість діагнозів (вихід)
            som_dim: Розмір SOM context (k юнітів)
            symptom_hidden: Розміри шарів гілки симптомів [256, 128]
            symptom_dropout: Dropout для гілки симптомів [0.3, 0.3]
            som_hidden: Розміри шарів гілки SOM [64, 32]
            som_dropout: Dropout для гілки SOM [0.2, 0.2]
            combined_hidden: Розміри об'єднаних шарів [128]
            combined_dropout: Dropout для об'єднаної частини
            use_batch_norm: Використовувати BatchNorm
            normalize_symptoms: L2-нормалізація вхідного вектора симптомів
        """
        super().__init__()
        
        self.n_symptoms = n_symptoms
        self.n_diseases = n_diseases
        self.som_dim = som_dim
        self.normalize_symptoms = normalize_symptoms
        
        # Defaults з CONFIG_PARAMETERS.md
        if symptom_hidden is None:
            symptom_hidden = [256, 128]
        if symptom_dropout is None:
            symptom_dropout = [0.3, 0.3]
        if som_hidden is None:
            som_hidden = [64, 32]
        if som_dropout is None:
            som_dropout = [0.2, 0.2]
        if combined_hidden is None:
            combined_hidden = [128]
        
        # ====== ГІЛКА 1: Симптоми ======
        self.symptom_branch = self._build_branch(
            input_dim=n_symptoms,
            hidden_dims=symptom_hidden,
            dropouts=symptom_dropout,
            use_batch_norm=use_batch_norm,
            name="symptom"
        )
        symptom_output_dim = symptom_hidden[-1]
        
        # ====== ГІЛКА 2: SOM Context ======
        self.som_branch = self._build_branch(
            input_dim=som_dim,
            hidden_dims=som_hidden,
            dropouts=som_dropout,
            use_batch_norm=use_batch_norm,
            name="som"
        )
        som_output_dim = som_hidden[-1]
        
        # ====== ОБ'ЄДНАНА ЧАСТИНА ======
        combined_input_dim = symptom_output_dim + som_output_dim
        
        combined_layers = []
        in_dim = combined_input_dim
        
        for out_dim in combined_hidden:
            combined_layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                combined_layers.append(nn.BatchNorm1d(out_dim))
            combined_layers.append(nn.ReLU())
            combined_layers.append(nn.Dropout(combined_dropout))
            in_dim = out_dim
        
        self.combined_layers = nn.Sequential(*combined_layers)
        
        # ====== ВИХІДНИЙ ШАР ======
        self.output_layer = nn.Linear(combined_hidden[-1], n_diseases)
        
        # Ініціалізація ваг
        self._init_weights()
        
        # Зберігаємо конфігурацію для serialization
        self.config = {
            "n_symptoms": n_symptoms,
            "n_diseases": n_diseases,
            "som_dim": som_dim,
            "symptom_hidden": symptom_hidden,
            "symptom_dropout": symptom_dropout,
            "som_hidden": som_hidden,
            "som_dropout": som_dropout,
            "combined_hidden": combined_hidden,
            "combined_dropout": combined_dropout,
            "use_batch_norm": use_batch_norm,
            "normalize_symptoms": normalize_symptoms,
        }
    
    def _build_branch(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropouts: List[float],
        use_batch_norm: bool,
        name: str
    ) -> nn.Sequential:
        """Побудувати одну гілку мережі"""
        layers = []
        in_dim = input_dim
        
        for i, (out_dim, dropout) in enumerate(zip(hidden_dims, dropouts)):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """He ініціалізація для ReLU"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        symptoms: torch.Tensor,
        som_context: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            symptoms: Вектор симптомів shape (batch, n_symptoms)
            som_context: SOM membership shape (batch, som_dim)
            candidate_mask: Маска кандидатів shape (batch, n_diseases)
                           1 = кандидат, 0 = не кандидат
                           
        Returns:
            Scores shape (batch, n_diseases)
        """
        # L2-нормалізація симптомів (якщо увімкнено)
        if self.normalize_symptoms:
            symptoms = F.normalize(symptoms, p=2, dim=-1)
        
        # Гілка симптомів
        h_symptoms = self.symptom_branch(symptoms)
        
        # Гілка SOM (membership вже нормалізовано через softmax)
        h_som = self.som_branch(som_context)
        
        # Об'єднання
        h_combined = torch.cat([h_symptoms, h_som], dim=-1)
        
        # Об'єднані шари
        h = self.combined_layers(h_combined)
        
        # Вихід
        scores = self.output_layer(h)
        
        # Маска кандидатів
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask == 0, float('-inf'))
        
        return scores
    
    def predict_proba(
        self,
        symptoms: torch.Tensor,
        som_context: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Отримати ймовірності (softmax)"""
        scores = self.forward(symptoms, som_context, candidate_mask)
        return F.softmax(scores, dim=-1)
    
    def get_top_k(
        self,
        symptoms: torch.Tensor,
        som_context: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Отримати топ-K діагнозів"""
        scores = self.forward(symptoms, som_context, candidate_mask)
        return torch.topk(scores, k=min(k, scores.size(-1)), dim=-1)
    
    def count_parameters(self) -> int:
        """Кількість параметрів моделі"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"TwoBranchNN(\n"
            f"  symptoms={self.n_symptoms} → {self.config['symptom_hidden']}\n"
            f"  som={self.som_dim} → {self.config['som_hidden']}\n"
            f"  combined → {self.config['combined_hidden']} → diseases={self.n_diseases}\n"
            f"  normalize_symptoms={self.normalize_symptoms}\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )


class TwoBranchDataset(torch.utils.data.Dataset):
    """
    Dataset для двогілкової архітектури.
    
    Кожен приклад: (symptom_vector, som_context, disease_index)
    """
    
    def __init__(
        self,
        symptom_vectors: np.ndarray,
        som_contexts: np.ndarray,
        disease_indices: np.ndarray,
        augment: bool = False,
        augment_prob: float = 0.3,
        drop_prob: float = 0.15,
        add_prob: float = 0.05,
    ):
        """
        Args:
            symptom_vectors: Матриця симптомів (N, D)
            som_contexts: Матриця SOM context (N, k)
            disease_indices: Індекси діагнозів (N,)
            augment: Аугментація симптомів
            augment_prob: Ймовірність аугментації
            drop_prob: Ймовірність видалення симптому
            add_prob: Ймовірність додавання симптому
        """
        self.symptom_vectors = torch.FloatTensor(symptom_vectors)
        self.som_contexts = torch.FloatTensor(som_contexts)
        self.disease_indices = torch.LongTensor(disease_indices)
        
        self.augment = augment
        self.augment_prob = augment_prob
        self.drop_prob = drop_prob
        self.add_prob = add_prob
        self.n_symptoms = symptom_vectors.shape[1]
    
    def __len__(self) -> int:
        return len(self.disease_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        symptoms = self.symptom_vectors[idx].clone()
        som_context = self.som_contexts[idx]
        target = self.disease_indices[idx]
        
        # Аугментація тільки симптомів (не SOM context!)
        if self.augment and np.random.random() < self.augment_prob:
            symptoms = self._augment_symptoms(symptoms)
        
        return symptoms, som_context, target
    
    def _augment_symptoms(self, vector: torch.Tensor) -> torch.Tensor:
        """Аугментація вектора симптомів"""
        active = (vector > 0).nonzero(as_tuple=True)[0]
        inactive = (vector == 0).nonzero(as_tuple=True)[0]
        
        # Dropout симптомів (залишаємо хоча б 2)
        if len(active) > 2:
            for idx in active:
                if np.random.random() < self.drop_prob:
                    vector[idx] = 0
        
        # Додавання "шуму"
        if len(inactive) > 0:
            n_add = np.random.binomial(len(inactive), self.add_prob)
            if n_add > 0:
                add_indices = np.random.choice(inactive.numpy(), n_add, replace=False)
                vector[add_indices] = 1
        
        return vector


def normalize_symptoms(symptoms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-нормалізація вектора симптомів.
    
    Args:
        symptoms: Вектор(и) симптомів shape (D,) або (N, D)
        eps: Epsilon для уникнення ділення на 0
        
    Returns:
        Нормалізований вектор(и)
    """
    if symptoms.ndim == 1:
        norm = np.linalg.norm(symptoms)
        return symptoms / (norm + eps)
    else:
        norms = np.linalg.norm(symptoms, axis=1, keepdims=True)
        return symptoms / (norms + eps)
