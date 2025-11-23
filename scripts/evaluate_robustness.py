"""
Noise Robustness Evaluation for Miyraa NLP Engine

Tests model robustness against various types of noise:
- Typos (character swaps, deletions, insertions)
- Slang and informal text
- Mixed languages (code-switching)
- Repeated characters
- Emoji insertion
- Special characters

Measures performance degradation across noise levels.

Usage:
    python scripts/evaluate_robustness.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap
    
    # Test specific noise types
    python scripts/evaluate_robustness.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --noise-types typos slang
    
    # Custom noise levels
    python scripts/evaluate_robustness.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --noise-levels 0.1 0.2 0.5

Author: Miyraa Team
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from datetime import datetime
import json
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.models.multi_task_model import MultiTaskModel


class NoiseInjector:
    """
    Inject various types of noise into text to test robustness.
    
    Example:
        >>> injector = NoiseInjector(noise_level=0.2)
        >>> noisy_text = injector.add_typos("Hello world")
        >>> noisy_text = injector.add_slang("I am very happy")
    """
    
    def __init__(self, noise_level: float = 0.2, seed: int = 42):
        """
        Initialize noise injector.
        
        Args:
            noise_level: Probability of applying noise (0 to 1)
            seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        random.seed(seed)
        np.random.seed(seed)
        
        # Common typo patterns
        self.keyboard_neighbors = {
            'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wrdsf',
            'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'uojkl', 'j': 'huikmn',
            'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'awedxz', 't': 'ryfg',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tugh',
            'z': 'asx'
        }
        
        # Slang replacements
        self.slang_map = {
            'you': 'u',
            'your': 'ur',
            'are': 'r',
            'for': '4',
            'to': '2',
            'too': '2',
            'see': 'c',
            'be': 'b',
            'okay': 'ok',
            'tonight': '2nite',
            'tomorrow': 'tmrw',
            'because': 'bc',
            'before': 'b4',
            'great': 'gr8',
            'later': 'l8r',
            'please': 'pls',
            'thanks': 'thx',
            'probably': 'prob',
            'definitely': 'def',
            'laugh': 'lol',
            'oh my god': 'omg',
            'by the way': 'btw',
            'in my opinion': 'imo'
        }
        
        # Common emojis
        self.emojis = ['😀', '😂', '😊', '😢', '😡', '😍', '👍', '👎', '🙏', '🔥', '💯', '❤️']
        
        # Mixed language fragments (common code-switching patterns)
        self.mixed_fragments = {
            'hello': ['hola', 'bonjour', 'hallo'],
            'thank you': ['gracias', 'merci', 'danke'],
            'yes': ['si', 'oui', 'ja'],
            'no': ['non', 'nein'],
            'good': ['bien', 'gut', 'bon'],
            'bad': ['mal', 'schlecht', 'mauvais']
        }
    
    def add_typos(self, text: str) -> str:
        """
        Add typos to text (character swaps, deletions, insertions).
        
        Args:
            text: Input text
        
        Returns:
            noisy_text: Text with typos
        """
        words = text.split()
        noisy_words = []
        
        for word in words:
            if len(word) < 3 or random.random() > self.noise_level:
                noisy_words.append(word)
                continue
            
            word_chars = list(word.lower())
            typo_type = random.choice(['swap', 'delete', 'insert', 'neighbor'])
            
            if typo_type == 'swap' and len(word_chars) >= 2:
                # Swap adjacent characters
                idx = random.randint(0, len(word_chars) - 2)
                word_chars[idx], word_chars[idx + 1] = word_chars[idx + 1], word_chars[idx]
            
            elif typo_type == 'delete' and len(word_chars) > 2:
                # Delete a character
                idx = random.randint(0, len(word_chars) - 1)
                word_chars.pop(idx)
            
            elif typo_type == 'insert':
                # Insert a random character
                idx = random.randint(0, len(word_chars))
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                word_chars.insert(idx, char)
            
            elif typo_type == 'neighbor':
                # Replace with keyboard neighbor
                idx = random.randint(0, len(word_chars) - 1)
                char = word_chars[idx]
                if char in self.keyboard_neighbors:
                    word_chars[idx] = random.choice(self.keyboard_neighbors[char])
            
            noisy_words.append(''.join(word_chars))
        
        return ' '.join(noisy_words)
    
    def add_slang(self, text: str) -> str:
        """
        Replace words with slang/informal equivalents.
        
        Args:
            text: Input text
        
        Returns:
            noisy_text: Text with slang
        """
        text_lower = text.lower()
        
        for formal, slang in self.slang_map.items():
            if random.random() < self.noise_level and formal in text_lower:
                # Replace with slang (case-insensitive)
                text = text.replace(formal, slang)
                text = text.replace(formal.capitalize(), slang.capitalize())
        
        return text
    
    def add_repeated_chars(self, text: str) -> str:
        """
        Add repeated characters (e.g., "sooo happy").
        
        Args:
            text: Input text
        
        Returns:
            noisy_text: Text with repeated characters
        """
        words = text.split()
        noisy_words = []
        
        for word in words:
            if len(word) < 3 or random.random() > self.noise_level:
                noisy_words.append(word)
                continue
            
            # Pick a random vowel to repeat
            vowels = [i for i, c in enumerate(word) if c.lower() in 'aeiou']
            if vowels:
                idx = random.choice(vowels)
                char = word[idx]
                # Repeat 2-4 times
                repeat_count = random.randint(2, 4)
                word = word[:idx] + char * repeat_count + word[idx+1:]
            
            noisy_words.append(word)
        
        return ' '.join(noisy_words)
    
    def add_emojis(self, text: str) -> str:
        """
        Insert random emojis into text.
        
        Args:
            text: Input text
        
        Returns:
            noisy_text: Text with emojis
        """
        if random.random() < self.noise_level:
            # Add 1-3 emojis
            num_emojis = random.randint(1, 3)
            emojis = random.choices(self.emojis, k=num_emojis)
            
            # Insert at random positions
            words = text.split()
            for emoji in emojis:
                if words:
                    idx = random.randint(0, len(words))
                    words.insert(idx, emoji)
            
            return ' '.join(words)
        
        return text
    
    def add_mixed_language(self, text: str) -> str:
        """
        Add mixed language fragments (code-switching).
        
        Args:
            text: Input text
        
        Returns:
            noisy_text: Text with mixed languages
        """
        text_lower = text.lower()
        
        for english, translations in self.mixed_fragments.items():
            if random.random() < self.noise_level and english in text_lower:
                # Replace with random translation
                translation = random.choice(translations)
                text = text.replace(english, translation)
                text = text.replace(english.capitalize(), translation.capitalize())
        
        return text
    
    def add_special_chars(self, text: str) -> str:
        """
        Add special characters and punctuation.
        
        Args:
            text: Input text
        
        Returns:
            noisy_text: Text with special characters
        """
        special_chars = ['!', '?', '...', '!!!', '???', '!?']
        
        if random.random() < self.noise_level:
            # Add special chars at end
            text = text.rstrip('.!?')
            text += random.choice(special_chars)
        
        return text
    
    def apply_noise(self, text: str, noise_types: List[str]) -> str:
        """
        Apply multiple noise types to text.
        
        Args:
            text: Input text
            noise_types: List of noise types to apply
        
        Returns:
            noisy_text: Text with all noise applied
        """
        for noise_type in noise_types:
            if noise_type == 'typos':
                text = self.add_typos(text)
            elif noise_type == 'slang':
                text = self.add_slang(text)
            elif noise_type == 'repeated':
                text = self.add_repeated_chars(text)
            elif noise_type == 'emojis':
                text = self.add_emojis(text)
            elif noise_type == 'mixed_lang':
                text = self.add_mixed_language(text)
            elif noise_type == 'special_chars':
                text = self.add_special_chars(text)
        
        return text


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load model checkpoint and tokenizer."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    metadata = checkpoint.get('metadata', {})
    backbone_name = metadata.get('backbone_name', 'xlm-roberta-base')
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    model = MultiTaskModel(
        backbone_name=backbone_name,
        num_emotion_classes=5,
        num_sentiment_classes=3,
        num_intent_classes=10,
        num_topic_classes=8,
        num_safety_classes=2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    
    return model, tokenizer


def evaluate_on_noisy_data(
    model,
    tokenizer,
    texts: List[str],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluate model on noisy text data.
    
    Returns:
        accuracy: Classification accuracy
        f1: Macro F1 score
    """
    y_true = []
    y_pred = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emotion_logits = outputs['emotion']
            preds = torch.argmax(emotion_logits, dim=-1).cpu().numpy()
        
        y_true.extend(batch_labels)
        y_pred.extend(preds)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description='Evaluate model robustness to noise')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='reports/robustness',
                        help='Output directory for reports')
    parser.add_argument('--noise-types', nargs='+',
                        default=['typos', 'slang', 'repeated', 'emojis', 'mixed_lang', 'special_chars'],
                        help='Types of noise to test')
    parser.add_argument('--noise-levels', type=float, nargs='+',
                        default=[0.1, 0.2, 0.5],
                        help='Noise levels to test')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples to test')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NOISE ROBUSTNESS EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Noise types: {args.noise_types}")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Samples: {args.n_samples}")
    print("="*60 + "\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    
    # Load dataset
    print(f"Loading validation data...")
    dataset_dict = load_from_disk(args.data)
    val_dataset = dataset_dict['validation'] if 'validation' in dataset_dict else dataset_dict['train']
    
    # Sample subset
    if len(val_dataset) > args.n_samples:
        indices = np.random.choice(len(val_dataset), args.n_samples, replace=False)
        val_dataset = val_dataset.select(indices)
    
    texts = val_dataset['text']
    labels = np.array(val_dataset['emotion'])
    
    print(f"✓ Loaded {len(texts)} samples\n")
    
    # Evaluate on clean data first
    print("Evaluating on clean data (baseline)...")
    clean_accuracy, clean_f1 = evaluate_on_noisy_data(
        model, tokenizer, texts, labels, device, args.batch_size
    )
    print(f"  Clean Accuracy: {clean_accuracy:.4f}")
    print(f"  Clean F1: {clean_f1:.4f}\n")
    
    # Test each noise type at each level
    results = {
        'clean': {'accuracy': clean_accuracy, 'f1': clean_f1}
    }
    
    for noise_type in args.noise_types:
        print(f"\nTesting noise type: {noise_type}")
        print("-" * 40)
        
        results[noise_type] = {}
        
        for noise_level in args.noise_levels:
            print(f"  Noise level: {noise_level:.1%}")
            
            # Apply noise
            injector = NoiseInjector(noise_level=noise_level, seed=args.seed)
            noisy_texts = [injector.apply_noise(text, [noise_type]) for text in texts]
            
            # Evaluate
            accuracy, f1 = evaluate_on_noisy_data(
                model, tokenizer, noisy_texts, labels, device, args.batch_size
            )
            
            results[noise_type][noise_level] = {
                'accuracy': accuracy,
                'f1': f1,
                'accuracy_drop': clean_accuracy - accuracy,
                'f1_drop': clean_f1 - f1
            }
            
            print(f"    Accuracy: {accuracy:.4f} (drop: {clean_accuracy - accuracy:.4f})")
            print(f"    F1: {f1:.4f} (drop: {clean_f1 - f1:.4f})")
    
    # Plot results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Plot 1: Accuracy degradation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for noise_type in args.noise_types:
        levels = args.noise_levels
        accuracies = [results[noise_type][level]['accuracy'] for level in levels]
        ax1.plot(levels, accuracies, 'o-', label=noise_type, linewidth=2, markersize=8)
    
    ax1.axhline(y=clean_accuracy, color='black', linestyle='--', label='Clean', linewidth=2)
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Noise Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 degradation
    for noise_type in args.noise_types:
        levels = args.noise_levels
        f1_scores = [results[noise_type][level]['f1'] for level in levels]
        ax2.plot(levels, f1_scores, 'o-', label=noise_type, linewidth=2, markersize=8)
    
    ax2.axhline(y=clean_f1, color='black', linestyle='--', label='Clean', linewidth=2)
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs Noise Level', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Robustness curves saved: robustness_curves.png")
    
    # Save results
    summary = {
        'checkpoint': args.checkpoint,
        'timestamp': timestamp,
        'n_samples': len(texts),
        'clean_performance': {
            'accuracy': clean_accuracy,
            'f1': clean_f1
        },
        'noise_types': args.noise_types,
        'noise_levels': args.noise_levels,
        'results': results
    }
    
    summary_path = output_dir / 'robustness_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - robustness_curves.png")
    print("  - robustness_summary.json")


if __name__ == "__main__":
    main()
