"""Enhanced multi-task model architecture with improved regularization and flexibility.

Features:
- Improved task heads with dropout and layer normalization
- Flexible backbone freezing strategies
- Residual connections for deeper heads
- Support for multiple backbone options
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel


class ImprovedTaskHead(nn.Module):
    """Enhanced task head with dropout, normalization, and residual connections.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (num_classes for classification, num_values for regression)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        use_layer_norm: Whether to use layer normalization
        use_residual: Whether to use residual connections
        task_type: 'classification' or 'regression'
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        task_type: str = 'classification'
    ):
        super().__init__()
        self.task_type = task_type
        self.use_residual = use_residual
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Residual projection if needed
        if use_residual and input_dim != hidden_dims[-1]:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        identity = x
        
        # Pass through hidden layers
        out = self.hidden_layers(x)
        
        # Add residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        # Output layer
        out = self.output_layer(out)
        
        return out


class MultiTaskModel(nn.Module):
    """Enhanced multi-task model with improved architecture.
    
    Args:
        backbone_name: HuggingFace model name
        num_emotions: Number of emotion classes
        num_safety_classes: Number of safety classes
        num_style_classes: Number of style classes
        num_intent_classes: Number of intent classes
        dropout_rate: Dropout probability for task heads
        freeze_backbone: Whether to freeze backbone weights
        freeze_layers: Number of backbone layers to freeze (from bottom)
        head_hidden_dims: Hidden dimensions for task heads
        use_layer_norm: Use layer normalization in heads
        use_residual: Use residual connections in heads
    """
    
    def __init__(
        self,
        backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_emotions: int = 11,
        num_safety_classes: int = 4,
        num_style_classes: int = 5,
        num_intent_classes: int = 6,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        freeze_layers: Optional[int] = None,
        head_hidden_dims: List[int] = [256, 128],
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.freeze_layers = freeze_layers
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # Apply freezing strategy
        self._apply_freezing()
        
        # Task heads with improved architecture
        self.emotion_head = ImprovedTaskHead(
            input_dim=self.hidden_size,
            output_dim=num_emotions,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            task_type='classification'
        )
        
        self.vad_head = ImprovedTaskHead(
            input_dim=self.hidden_size,
            output_dim=3,  # valence, arousal, dominance
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            task_type='regression'
        )
        
        self.safety_head = ImprovedTaskHead(
            input_dim=self.hidden_size,
            output_dim=num_safety_classes,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            task_type='classification'
        )
        
        self.style_head = ImprovedTaskHead(
            input_dim=self.hidden_size,
            output_dim=num_style_classes,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            task_type='classification'
        )
        
        self.intent_head = ImprovedTaskHead(
            input_dim=self.hidden_size,
            output_dim=num_intent_classes,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            task_type='classification'
        )
    
    def _apply_freezing(self):
        """Apply backbone freezing strategy."""
        if self.freeze_backbone:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✅ Backbone fully frozen")
        
        elif self.freeze_layers is not None:
            # Freeze specific number of layers
            if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
                # BERT-style models
                layers = self.backbone.encoder.layer
                num_layers = len(layers)
                freeze_count = min(self.freeze_layers, num_layers)
                
                for i in range(freeze_count):
                    for param in layers[i].parameters():
                        param.requires_grad = False
                
                print(f"✅ Frozen {freeze_count}/{num_layers} backbone layers")
            else:
                print("⚠️ Could not freeze specific layers (model structure unknown)")
        else:
            print("✅ Backbone fully unfrozen")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all tasks.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with keys: emotions, vad, safety, style, intent
        """
        # Get backbone embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Pass through task heads
        return {
            'emotions': self.emotion_head(embeddings),
            'vad': self.vad_head(embeddings),
            'safety': self.safety_head(embeddings),
            'style': self.style_head(embeddings),
            'intent': self.intent_head(embeddings)
        }
    
    def get_trainable_params(self) -> Tuple[int, int]:
        """Get count of trainable and total parameters.
        
        Returns:
            (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def unfreeze_layers(self, num_layers: int):
        """Unfreeze top N layers of backbone during training.
        
        Args:
            num_layers: Number of layers to unfreeze from top
        """
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer
            num_layers_to_unfreeze = min(num_layers, len(layers))
            
            # Unfreeze from top
            for i in range(len(layers) - num_layers_to_unfreeze, len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
            
            print(f"✅ Unfroze top {num_layers_to_unfreeze} backbone layers")
        else:
            print("⚠️ Could not unfreeze layers (model structure unknown)")


def create_model(
    backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    freeze_strategy: str = "none",  # "none", "full", "partial"
    freeze_layers: int = 4,
    dropout_rate: float = 0.3,
    head_hidden_dims: List[int] = [256, 128],
    **kwargs
) -> MultiTaskModel:
    """Factory function to create model with specified configuration.
    
    Args:
        backbone_name: HuggingFace model name
        freeze_strategy: Freezing strategy ("none", "full", "partial")
        freeze_layers: Number of layers to freeze if partial
        dropout_rate: Dropout rate for heads
        head_hidden_dims: Hidden dimensions for heads
        **kwargs: Additional arguments for MultiTaskModel
        
    Returns:
        Configured MultiTaskModel
    """
    freeze_backbone = (freeze_strategy == "full")
    freeze_layers_arg = freeze_layers if freeze_strategy == "partial" else None
    
    model = MultiTaskModel(
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers_arg,
        dropout_rate=dropout_rate,
        head_hidden_dims=head_hidden_dims,
        **kwargs
    )
    
    trainable, total = model.get_trainable_params()
    print(f"\n📊 Model Parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Percentage: {100 * trainable / total:.1f}%")
    
    return model


if __name__ == "__main__":
    # Test model creation with different configurations
    print("=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)
    
    # Test 1: Fully unfrozen
    print("\n1. Fully Unfrozen Backbone:")
    model1 = create_model(freeze_strategy="none")
    
    # Test 2: Fully frozen
    print("\n2. Fully Frozen Backbone:")
    model2 = create_model(freeze_strategy="full")
    
    # Test 3: Partially frozen
    print("\n3. Partially Frozen (4 layers):")
    model3 = create_model(freeze_strategy="partial", freeze_layers=4)
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model1(input_ids, attention_mask)
    
    print(f"\nOutput shapes:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    print("\n✅ All tests passed!")
