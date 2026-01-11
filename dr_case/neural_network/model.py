"""
Dr.Case — Архітектура нейронної мережі

Neural Network для ранжування діагнозів-кандидатів.

Архітектура:
- Input: вектор симптомів (461) + embedding діагнозу
- Hidden layers: FC з BatchNorm та Dropout
- Output: score для кожного кандидата

Підтримує два режими:
1. Pointwise: окремий score для кожного (symptom, disease) pair
2. Listwise: scores для всіх кандидатів одночасно
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class DiagnosisNN(nn.Module):
    """
    Нейронна мережа для ранжування діагнозів.
    
    Приймає вектор симптомів та повертає scores для всіх діагнозів.
    При інференсі використовується маска кандидатів.
    
    Приклад:
        model = DiagnosisNN(
            n_symptoms=461,
            n_diseases=842,
            hidden_dims=[512, 256, 128]
        )
        
        # Forward
        symptoms = torch.randn(batch_size, 461)
        scores = model(symptoms)  # shape (batch_size, 842)
        
        # З маскою кандидатів
        candidate_mask = torch.zeros(batch_size, 842)
        candidate_mask[:, candidate_indices] = 1
        masked_scores = model(symptoms, candidate_mask)
    """
    
    def __init__(
        self,
        n_symptoms: int = 461,
        n_diseases: int = 842,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            n_symptoms: Кількість симптомів (розмір входу)
            n_diseases: Кількість діагнозів (розмір виходу)
            hidden_dims: Розміри прихованих шарів
            dropout: Dropout rate
            use_batch_norm: Чи використовувати BatchNorm
        """
        super().__init__()
        
        self.n_symptoms = n_symptoms
        self.n_diseases = n_diseases
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Будуємо шари
        layers = []
        in_dim = n_symptoms
        
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Вихідний шар
        self.output_layer = nn.Linear(hidden_dims[-1], n_diseases)
        
        # Ініціалізація ваг
        self._init_weights()
    
    def _init_weights(self):
        """Xavier ініціалізація"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        symptoms: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            symptoms: Вектор симптомів shape (batch, n_symptoms)
            candidate_mask: Маска кандидатів shape (batch, n_diseases)
                           1 = кандидат, 0 = не кандидат
                           
        Returns:
            Scores shape (batch, n_diseases)
        """
        # Приховані шари
        h = self.hidden_layers(symptoms)
        
        # Вихідний шар
        scores = self.output_layer(h)
        
        # Застосовуємо маску (якщо є)
        if candidate_mask is not None:
            # Встановлюємо -inf для не-кандидатів
            scores = scores.masked_fill(candidate_mask == 0, float('-inf'))
        
        return scores
    
    def predict_proba(
        self,
        symptoms: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Отримати ймовірності (softmax).
        
        Args:
            symptoms: Вектор симптомів
            candidate_mask: Маска кандидатів
            
        Returns:
            Probabilities shape (batch, n_diseases)
        """
        scores = self.forward(symptoms, candidate_mask)
        return F.softmax(scores, dim=-1)
    
    def get_top_k(
        self,
        symptoms: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Отримати топ-K діагнозів.
        
        Args:
            symptoms: Вектор симптомів
            candidate_mask: Маска кандидатів
            k: Кількість топ результатів
            
        Returns:
            (top_scores, top_indices) обидва shape (batch, k)
        """
        scores = self.forward(symptoms, candidate_mask)
        return torch.topk(scores, k=min(k, scores.size(-1)), dim=-1)
    
    def count_parameters(self) -> int:
        """Кількість параметрів моделі"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"DiagnosisNN(\n"
            f"  symptoms={self.n_symptoms}, diseases={self.n_diseases},\n"
            f"  hidden={self.hidden_dims}, dropout={self.dropout}\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )


class DiagnosisNNWithEmbedding(nn.Module):
    """
    Альтернативна архітектура з embedding діагнозів.
    
    Замість одного великого виходу, використовує:
    - Encoder для симптомів → symptom_embedding
    - Embedding для діагнозів → disease_embedding  
    - Score = dot_product(symptom_emb, disease_emb)
    
    Це ефективніше для великої кількості діагнозів.
    """
    
    def __init__(
        self,
        n_symptoms: int = 461,
        n_diseases: int = 842,
        embedding_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_symptoms = n_symptoms
        self.n_diseases = n_diseases
        self.embedding_dim = embedding_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Encoder для симптомів
        encoder_layers = []
        in_dim = n_symptoms
        
        for out_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        encoder_layers.append(nn.Linear(in_dim, embedding_dim))
        self.symptom_encoder = nn.Sequential(*encoder_layers)
        
        # Embedding для діагнозів
        self.disease_embedding = nn.Embedding(n_diseases, embedding_dim)
        
        # Ініціалізація
        nn.init.xavier_uniform_(self.disease_embedding.weight)
    
    def forward(
        self,
        symptoms: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            symptoms: shape (batch, n_symptoms)
            candidate_indices: shape (batch, n_candidates) - індекси кандидатів
            
        Returns:
            Scores shape (batch, n_candidates) або (batch, n_diseases)
        """
        # Encode symptoms
        symptom_emb = self.symptom_encoder(symptoms)  # (batch, emb_dim)
        symptom_emb = F.normalize(symptom_emb, dim=-1)
        
        if candidate_indices is not None:
            # Тільки для кандидатів
            disease_emb = self.disease_embedding(candidate_indices)  # (batch, n_cand, emb_dim)
            disease_emb = F.normalize(disease_emb, dim=-1)
            
            # Dot product
            scores = torch.bmm(disease_emb, symptom_emb.unsqueeze(-1)).squeeze(-1)
        else:
            # Для всіх діагнозів
            all_disease_emb = self.disease_embedding.weight  # (n_diseases, emb_dim)
            all_disease_emb = F.normalize(all_disease_emb, dim=-1)
            
            scores = torch.mm(symptom_emb, all_disease_emb.t())  # (batch, n_diseases)
        
        return scores
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """
    Focal Loss для незбалансованих класів.
    
    FL(p) = -α * (1-p)^γ * log(p)
    
    Фокусується на складних прикладах.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits shape (batch, n_classes)
            targets: Target indices shape (batch,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_model(
    n_symptoms: int = 461,
    n_diseases: int = 842,
    architecture: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory функція для створення моделі.
    
    Args:
        n_symptoms: Кількість симптомів
        n_diseases: Кількість діагнозів
        architecture: "standard" або "embedding"
        **kwargs: Додаткові параметри
        
    Returns:
        nn.Module
    """
    if architecture == "standard":
        return DiagnosisNN(n_symptoms, n_diseases, **kwargs)
    elif architecture == "embedding":
        return DiagnosisNNWithEmbedding(n_symptoms, n_diseases, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
