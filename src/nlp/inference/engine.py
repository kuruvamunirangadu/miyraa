"""
Production Inference Engine for Miyraa NLP Multi-Task Emotion Model

Provides clean inference interface with:
- ONNX runtime support for fast inference
- PyTorch model support for flexibility
- Batch inference capabilities
- Label decoding and result formatting

Separated from training code for clean deployment.

Author: Miyraa Team
Date: November 2025
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import json

# Try ONNX runtime (optional)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from transformers import AutoTokenizer


class EmotionInferenceEngine:
    """
    Inference engine for multi-task emotion classification.
    
    Supports both PyTorch and ONNX models for flexible deployment.
    
    Example:
        >>> engine = EmotionInferenceEngine(
        ...     model_path='outputs/model.onnx',
        ...     model_type='onnx'
        ... )
        >>> result = engine.predict("I love this product!")
        >>> print(result['emotion'])  # 'joy'
    """
    
    # Label mappings
    EMOTION_LABELS = [
        'joy', 'love', 'surprise', 'sadness', 'anger',
        'fear', 'disgust', 'calm', 'excitement', 'confusion', 'neutral'
    ]
    
    STYLE_LABELS = ['formal', 'casual', 'assertive', 'empathetic', 'humorous']
    INTENT_LABELS = ['statement', 'question', 'request', 'command', 'expression', 'social']
    SAFETY_LABELS = ['safe', 'toxic', 'profane', 'threatening']
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'onnx',
        tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: str = 'cpu',
        max_length: int = 128
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file (.pt for PyTorch, .onnx for ONNX)
            model_type: 'pytorch' or 'onnx'
            tokenizer_name: HuggingFace tokenizer name
            device: 'cpu' or 'cuda'
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model
        if self.model_type == 'onnx':
            self._load_onnx_model()
        elif self.model_type == 'pytorch':
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'onnx' or 'pytorch'")
        
        print(f"✓ Inference engine initialized ({self.model_type})")
    
    def _load_onnx_model(self):
        """Load ONNX model with ONNX Runtime"""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime not installed. Install with: pip install onnxruntime"
            )
        
        print(f"Loading ONNX model: {self.model_path}")
        
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create inference session
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"✓ ONNX model loaded")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        print(f"Loading PyTorch model: {self.model_path}")
        
        # Import here to avoid circular dependency
        from src.nlp.models import create_model
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model (assuming default architecture)
        self.model = create_model(freeze_strategy='none')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ PyTorch model loaded")
    
    def preprocess(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Preprocess text(s) for inference.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            inputs: Dictionary of input arrays (input_ids, attention_mask)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'  # Return numpy arrays
        )
        
        return {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64),
        }
    
    def _predict_onnx(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run ONNX inference"""
        # Prepare inputs for ONNX
        onnx_inputs = {
            self.input_names[0]: inputs['input_ids'],
            self.input_names[1]: inputs['attention_mask'],
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, onnx_inputs)
        
        # Map outputs to names
        output_dict = {
            'emotions': outputs[0],
            'vad': outputs[1],
            'style': outputs[2],
            'intent': outputs[3],
            'safety': outputs[4],
        }
        
        return output_dict
    
    def _predict_pytorch(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run PyTorch inference"""
        # Convert to tensors
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Convert to numpy
        output_dict = {
            'emotions': outputs['emotions'].cpu().numpy(),
            'vad': outputs['vad'].cpu().numpy(),
            'style': outputs['style'].cpu().numpy(),
            'intent': outputs['intent'].cpu().numpy(),
            'safety': outputs['safety'].cpu().numpy(),
        }
        
        return output_dict
    
    def postprocess(self, outputs: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Convert model outputs to human-readable results.
        
        Args:
            outputs: Dictionary of model outputs (logits/values)
        
        Returns:
            results: List of result dictionaries (one per sample)
        """
        batch_size = outputs['emotions'].shape[0]
        results = []
        
        for i in range(batch_size):
            # Emotion classification
            emotion_logits = outputs['emotions'][i]
            emotion_id = int(np.argmax(emotion_logits))
            emotion_scores = self._softmax(emotion_logits)
            
            # VAD regression
            vad = outputs['vad'][i]
            
            # Style classification
            style_logits = outputs['style'][i]
            style_id = int(np.argmax(style_logits))
            
            # Intent classification
            intent_logits = outputs['intent'][i]
            intent_id = int(np.argmax(intent_logits))
            
            # Safety classification
            safety_logits = outputs['safety'][i]
            safety_id = int(np.argmax(safety_logits))
            
            result = {
                'emotion': self.EMOTION_LABELS[emotion_id],
                'emotion_id': emotion_id,
                'emotion_scores': {
                    label: float(score)
                    for label, score in zip(self.EMOTION_LABELS, emotion_scores)
                },
                'vad': {
                    'valence': float(vad[0]),
                    'arousal': float(vad[1]),
                    'dominance': float(vad[2]),
                },
                'style': self.STYLE_LABELS[style_id],
                'style_id': style_id,
                'intent': self.INTENT_LABELS[intent_id],
                'intent_id': intent_id,
                'safety': self.SAFETY_LABELS[safety_id],
                'safety_id': safety_id,
            }
            
            results.append(result)
        
        return results
    
    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Run inference on text(s).
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            result: Single result dict if input is string, list of dicts if list
        
        Example:
            >>> result = engine.predict("I'm so happy!")
            >>> print(result['emotion'])  # 'joy'
            
            >>> results = engine.predict(["Happy!", "Sad..."])
            >>> print([r['emotion'] for r in results])  # ['joy', 'sadness']
        """
        single_input = isinstance(texts, str)
        
        # Preprocess
        inputs = self.preprocess(texts)
        
        # Run inference
        if self.model_type == 'onnx':
            outputs = self._predict_onnx(inputs)
        else:
            outputs = self._predict_pytorch(inputs)
        
        # Postprocess
        results = self.postprocess(outputs)
        
        # Return single result or list
        return results[0] if single_input else results
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Run inference on large batch of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
        
        Returns:
            results: List of result dictionaries
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.predict(batch_texts)
            all_results.extend(batch_results)
        
        return all_results
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def load_engine(
    model_path: str,
    model_type: str = 'onnx',
    **kwargs
) -> EmotionInferenceEngine:
    """
    Convenience function to load inference engine.
    
    Args:
        model_path: Path to model file
        model_type: 'onnx' or 'pytorch'
        **kwargs: Additional arguments for EmotionInferenceEngine
    
    Returns:
        engine: Initialized inference engine
    
    Example:
        >>> engine = load_engine('outputs/model.onnx')
        >>> result = engine.predict("Great day!")
    """
    return EmotionInferenceEngine(model_path, model_type, **kwargs)


if __name__ == "__main__":
    """
    Test inference engine
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python engine.py <model_path> [model_type]")
        print("\nExample:")
        print("  python engine.py outputs/model.onnx onnx")
        print("  python engine.py outputs/checkpoint.pt pytorch")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'onnx'
    
    print(f"Testing Inference Engine")
    print(f"Model: {model_path}")
    print(f"Type: {model_type}\n")
    
    # Load engine
    engine = load_engine(model_path, model_type)
    
    # Test samples
    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience ever. So disappointing.",
        "Wow, I didn't expect that at all!",
        "I'm feeling calm and peaceful today.",
        "Help! This is an emergency!",
    ]
    
    print("\nRunning inference on test samples:\n")
    
    for text in test_texts:
        result = engine.predict(text)
        print(f"Text: {text}")
        print(f"  Emotion: {result['emotion']} (confidence: {result['emotion_scores'][result['emotion']]:.3f})")
        print(f"  VAD: V={result['vad']['valence']:.2f}, A={result['vad']['arousal']:.2f}, D={result['vad']['dominance']:.2f}")
        print(f"  Style: {result['style']}, Intent: {result['intent']}, Safety: {result['safety']}")
        print()
    
    print("✅ Inference engine test complete!")
