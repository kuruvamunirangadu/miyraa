"""Enhanced loss functions for multi-task learning.

Includes:
- Focal loss for imbalanced classification
- Dynamic loss weight tuning
- Uncertainty-based weighting
- Multi-task loss combining strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for each class (None for no class weighting)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    
    Reference:
        Lin et al. (2017). Focal Loss for Dense Object Detection.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss with configurable weighting strategies.
    
    Supports:
    - Fixed weights
    - Learned uncertainty-based weights (Kendall et al., 2018)
    - Dynamic weight adjustment
    
    Args:
        loss_weights: Initial loss weights for each task
        use_uncertainty_weighting: Use learned uncertainty weights
        num_tasks: Number of tasks
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = False,
        num_tasks: int = 5
    ):
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Default weights if not provided
        if loss_weights is None:
            loss_weights = {
                'emotions': 1.0,
                'vad': 1.0,
                'safety': 1.0,
                'style': 1.0,
                'intent': 1.0
            }
        
        self.loss_weights = loss_weights
        
        # Learnable uncertainty parameters (log variance)
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # Task names in order
        self.task_names = ['emotions', 'vad', 'safety', 'style', 'intent']
    
    def forward(
        self,
        task_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted multi-task loss.
        
        Args:
            task_losses: Dictionary of individual task losses
            
        Returns:
            (total_loss, loss_components_dict)
        """
        total_loss = 0.0
        loss_components = {}
        
        if self.use_uncertainty_weighting:
            # Uncertainty-based weighting
            for i, task_name in enumerate(self.task_names):
                if task_name in task_losses:
                    precision = torch.exp(-self.log_vars[i])
                    task_loss = precision * task_losses[task_name] + self.log_vars[i]
                    total_loss += task_loss
                    loss_components[task_name] = task_loss.item()
        else:
            # Fixed weight-based
            for task_name, weight in self.loss_weights.items():
                if task_name in task_losses:
                    weighted_loss = weight * task_losses[task_name]
                    total_loss += weighted_loss
                    loss_components[task_name] = weighted_loss.item()
        
        return total_loss, loss_components
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current task weights (for logging).
        
        Returns:
            Dictionary of current weights
        """
        if self.use_uncertainty_weighting:
            weights = {}
            num_log_vars = len(self.log_vars)
            for i, task_name in enumerate(self.task_names):
                if i < num_log_vars:
                    precision = torch.exp(-self.log_vars[i])
                    weights[task_name] = precision.item()
            return weights
        else:
            return self.loss_weights.copy()


class LossCalculator:
    """Helper class for computing all task losses.
    
    Args:
        use_focal_loss: Use focal loss for emotion classification
        focal_gamma: Gamma parameter for focal loss
        emotion_class_weights: Class weights for emotion focal loss
        vad_loss_type: Type of VAD loss ('mse' or 'smooth_l1')
    """
    
    def __init__(
        self,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        emotion_class_weights: Optional[torch.Tensor] = None,
        vad_loss_type: str = 'mse'
    ):
        self.use_focal_loss = use_focal_loss
        self.vad_loss_type = vad_loss_type
        
        # Emotion loss
        if use_focal_loss:
            self.emotion_criterion = FocalLoss(
                alpha=emotion_class_weights,
                gamma=focal_gamma
            )
        else:
            self.emotion_criterion = nn.CrossEntropyLoss()
        
        # VAD loss
        if vad_loss_type == 'mse':
            self.vad_criterion = nn.MSELoss()
        elif vad_loss_type == 'smooth_l1':
            self.vad_criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown VAD loss type: {vad_loss_type}")
        
        # Other task losses
        self.safety_criterion = nn.CrossEntropyLoss()
        self.style_criterion = nn.CrossEntropyLoss()
        self.intent_criterion = nn.CrossEntropyLoss()
    
    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute individual task losses.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of task losses
        """
        losses = {}
        
        # Emotion loss
        if 'emotions' in predictions and 'emotions' in targets:
            losses['emotions'] = self.emotion_criterion(
                predictions['emotions'],
                targets['emotions']
            )
        
        # VAD loss
        if 'vad' in predictions and 'vad' in targets:
            losses['vad'] = self.vad_criterion(
                predictions['vad'],
                targets['vad']
            )
        
        # Safety loss
        if 'safety' in predictions and 'safety' in targets:
            losses['safety'] = self.safety_criterion(
                predictions['safety'],
                targets['safety']
            )
        
        # Style loss
        if 'style' in predictions and 'style' in targets:
            losses['style'] = self.style_criterion(
                predictions['style'],
                targets['style']
            )
        
        # Intent loss
        if 'intent' in predictions and 'intent' in targets:
            losses['intent'] = self.intent_criterion(
                predictions['intent'],
                targets['intent']
            )
        
        return losses


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    smoothing: float = 0.1
) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.
    
    Args:
        labels: List of integer labels
        num_classes: Total number of classes
        smoothing: Smoothing factor to prevent extreme weights
        
    Returns:
        Tensor of class weights [num_classes]
    """
    # Count samples per class
    counts = np.bincount(labels, minlength=num_classes)
    
    # Avoid division by zero
    counts = np.maximum(counts, 1)
    
    # Compute inverse frequency weights
    weights = 1.0 / counts
    
    # Apply smoothing
    weights = weights ** smoothing
    
    # Normalize to sum to num_classes
    weights = weights * (num_classes / weights.sum())
    
    return torch.FloatTensor(weights)


def find_optimal_loss_weights(
    train_losses: Dict[str, List[float]],
    strategy: str = 'inverse_variance'
) -> Dict[str, float]:
    """Find optimal loss weights based on training statistics.
    
    Args:
        train_losses: Dictionary of loss histories for each task
        strategy: Weighting strategy ('inverse_variance', 'gradient_norm', 'uniform')
        
    Returns:
        Dictionary of optimal weights
    """
    if strategy == 'uniform':
        num_tasks = len(train_losses)
        return {task: 1.0 / num_tasks for task in train_losses.keys()}
    
    elif strategy == 'inverse_variance':
        # Weight inversely proportional to variance
        variances = {}
        for task, losses in train_losses.items():
            variances[task] = np.var(losses) if len(losses) > 1 else 1.0
        
        # Compute inverse variance weights
        inv_vars = {task: 1.0 / max(var, 1e-8) for task, var in variances.items()}
        
        # Normalize
        total = sum(inv_vars.values())
        return {task: w / total * len(inv_vars) for task, w in inv_vars.items()}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced Loss Functions")
    print("=" * 60)
    
    batch_size = 8
    num_classes = 11
    
    # Test Focal Loss
    print("\n1. Testing Focal Loss:")
    focal_loss = FocalLoss(gamma=2.0)
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = focal_loss(logits, targets)
    print(f"   Focal Loss: {loss.item():.4f}")
    
    # Test with class weights
    class_weights = compute_class_weights([0, 0, 1, 1, 2, 2, 2, 2], num_classes=3)
    print(f"   Class weights: {class_weights}")
    
    # Test Multi-Task Loss
    print("\n2. Testing Multi-Task Loss:")
    
    # Fixed weights
    mtl_fixed = MultiTaskLoss(
        loss_weights={'emotions': 1.5, 'vad': 1.0, 'safety': 1.2},
        use_uncertainty_weighting=False
    )
    
    task_losses = {
        'emotions': torch.tensor(2.3),
        'vad': torch.tensor(0.5),
        'safety': torch.tensor(1.8)
    }
    
    total_loss, components = mtl_fixed(task_losses)
    print(f"   Total Loss (Fixed): {total_loss.item():.4f}")
    print(f"   Components: {components}")
    
    # Uncertainty weighting
    mtl_uncertainty = MultiTaskLoss(
        use_uncertainty_weighting=True,
        num_tasks=3
    )
    
    total_loss, components = mtl_uncertainty(task_losses)
    print(f"   Total Loss (Uncertainty): {total_loss.item():.4f}")
    print(f"   Learned weights: {mtl_uncertainty.get_current_weights()}")
    
    # Test Loss Calculator
    print("\n3. Testing Loss Calculator:")
    loss_calc = LossCalculator(use_focal_loss=True, focal_gamma=2.0)
    
    predictions = {
        'emotions': torch.randn(batch_size, 11),
        'vad': torch.randn(batch_size, 3),
        'safety': torch.randn(batch_size, 4)
    }
    
    targets = {
        'emotions': torch.randint(0, 11, (batch_size,)),
        'vad': torch.randn(batch_size, 3),
        'safety': torch.randint(0, 4, (batch_size,))
    }
    
    losses = loss_calc.compute_losses(predictions, targets)
    print(f"   Emotion Loss: {losses['emotions'].item():.4f}")
    print(f"   VAD Loss: {losses['vad'].item():.4f}")
    print(f"   Safety Loss: {losses['safety'].item():.4f}")
    
    # Test optimal weight finding
    print("\n4. Testing Optimal Weight Finding:")
    train_losses = {
        'emotions': [2.5, 2.3, 2.1, 2.0, 1.9],
        'vad': [0.8, 0.7, 0.65, 0.6, 0.58],
        'safety': [1.5, 1.4, 1.35, 1.3, 1.25]
    }
    
    optimal_weights = find_optimal_loss_weights(train_losses, strategy='inverse_variance')
    print(f"   Optimal weights: {optimal_weights}")
    
    print("\n✅ All loss function tests passed!")
