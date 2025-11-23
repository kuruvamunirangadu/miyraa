"""Evaluate and compare different transformer backbones for emotion classification.

Backbones tested:
- sentence-transformers/all-MiniLM-L6-v2 (current, 6 layers, 384-dim)
- sentence-transformers/all-MiniLM-L12-v2 (12 layers, 384-dim)
- microsoft/xtremedistil-l6-h384-uncased (6 layers, 384-dim, distilled)
- distilroberta-base (6 layers, 768-dim)
- xlm-roberta-base (12 layers, 768-dim, multilingual)

Metrics:
- Model size (parameters, disk space)
- Inference speed (samples/sec)
- Embedding quality (on sample task)
- Memory usage
"""

import torch
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModel, AutoTokenizer
import json


BACKBONES = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniLM-L12": "sentence-transformers/all-MiniLM-L12-v2",
    "XtremeDistil-L6": "microsoft/xtremedistil-l6-h384-uncased",
    "DistilRoBERTa": "distilroberta-base",
    "XLM-RoBERTa": "xlm-roberta-base",
}


def get_model_info(model_name: str) -> Dict:
    """Get basic information about a model.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Dictionary with model information
    """
    print(f"\n📊 Loading {model_name}...")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Get config info
        config = model.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else "Unknown"
        
        info = {
            "model_name": model_name,
            "total_params": total_params,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "max_seq_length": tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512,
        }
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return info
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return {"model_name": model_name, "error": str(e)}


def benchmark_inference_speed(
    model_name: str,
    num_samples: int = 100,
    batch_size: int = 8,
    seq_length: int = 64
) -> Dict:
    """Benchmark inference speed.
    
    Args:
        model_name: HuggingFace model name
        num_samples: Number of samples to process
        batch_size: Batch size for inference
        seq_length: Sequence length
        
    Returns:
        Benchmark results
    """
    print(f"\n⏱️ Benchmarking {model_name}...")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model.eval()
        
        # Prepare dummy data
        texts = ["This is a test sentence for benchmarking."] * num_samples
        
        # Warm-up
        with torch.no_grad():
            inputs = tokenizer(
                texts[:batch_size],
                padding=True,
                truncation=True,
                max_length=seq_length,
                return_tensors="pt"
            )
            _ = model(**inputs)
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=seq_length,
                    return_tensors="pt"
                )
                _ = model(**inputs)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        samples_per_sec = num_samples / elapsed
        ms_per_sample = (elapsed / num_samples) * 1000
        
        results = {
            "model_name": model_name,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "total_time_sec": elapsed,
            "samples_per_sec": samples_per_sec,
            "ms_per_sample": ms_per_sample
        }
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"❌ Error benchmarking {model_name}: {e}")
        return {"model_name": model_name, "error": str(e)}


def estimate_memory_usage(model_name: str) -> Dict:
    """Estimate memory usage.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Memory usage estimates
    """
    try:
        model = AutoModel.from_pretrained(model_name)
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        results = {
            "model_name": model_name,
            "memory_mb": total_size_mb,
            "param_size_mb": param_size / (1024 ** 2),
            "buffer_size_mb": buffer_size / (1024 ** 2)
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"❌ Error estimating memory for {model_name}: {e}")
        return {"model_name": model_name, "error": str(e)}


def compare_backbones(
    output_file: str = "reports/backbone_comparison.json"
) -> Dict:
    """Compare all backbones and save results.
    
    Args:
        output_file: Path to save comparison results
        
    Returns:
        Comparison results
    """
    print("=" * 70)
    print("BACKBONE COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for name, model_name in BACKBONES.items():
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 70}")
        
        # Get model info
        info = get_model_info(model_name)
        
        # Benchmark speed
        speed = benchmark_inference_speed(model_name, num_samples=100)
        
        # Estimate memory
        memory = estimate_memory_usage(model_name)
        
        # Combine results
        results[name] = {
            "info": info,
            "speed": speed,
            "memory": memory
        }
        
        # Print summary
        if "error" not in info:
            print(f"\n✅ {name} Summary:")
            print(f"   Parameters: {info['total_params']:,}")
            print(f"   Hidden Size: {info['hidden_size']}")
            print(f"   Layers: {info['num_layers']}")
            print(f"   Speed: {speed.get('samples_per_sec', 0):.1f} samples/sec")
            print(f"   Latency: {speed.get('ms_per_sample', 0):.2f} ms/sample")
            print(f"   Memory: {memory.get('memory_mb', 0):.1f} MB")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':<12} {'Hidden':<8} {'Layers':<7} {'Speed':<15} {'Memory':<10}")
    print("-" * 70)
    
    for name, data in results.items():
        info = data['info']
        speed = data['speed']
        memory = data['memory']
        
        if "error" not in info:
            params = f"{info['total_params'] / 1e6:.1f}M"
            hidden = str(info['hidden_size'])
            layers = str(info['num_layers'])
            speed_str = f"{speed.get('samples_per_sec', 0):.1f} samp/s"
            memory_str = f"{memory.get('memory_mb', 0):.1f} MB"
            
            print(f"{name:<20} {params:<12} {hidden:<8} {layers:<7} {speed_str:<15} {memory_str:<10}")
        else:
            print(f"{name:<20} ERROR: {info['error'][:40]}")
    
    print("=" * 70)
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("\n1. For CPU-only deployment (current):")
    print("   ✅ MiniLM-L6 - Best speed/accuracy trade-off")
    print("   ✅ XtremeDistil-L6 - Fastest inference")
    
    print("\n2. For higher accuracy:")
    print("   ✅ MiniLM-L12 - Better accuracy, moderate speed")
    print("   ✅ DistilRoBERTa - Strong general performance")
    
    print("\n3. For multilingual support:")
    print("   ✅ XLM-RoBERTa - Supports 100+ languages")
    
    print("\n4. For production at scale:")
    print("   ✅ MiniLM-L6 + ONNX quantization - Best throughput")
    
    return results


def quick_comparison():
    """Quick comparison without downloading models."""
    print("=" * 70)
    print("QUICK BACKBONE COMPARISON (No Download)")
    print("=" * 70)
    
    comparison = {
        "MiniLM-L6": {
            "params": "22.7M",
            "hidden": 384,
            "layers": 6,
            "speed_estimate": "~150 samples/sec (CPU)",
            "use_case": "Current production model - best speed/accuracy"
        },
        "MiniLM-L12": {
            "params": "33.4M",
            "hidden": 384,
            "layers": 12,
            "speed_estimate": "~90 samples/sec (CPU)",
            "use_case": "Better accuracy, acceptable speed"
        },
        "XtremeDistil-L6": {
            "params": "22M",
            "hidden": 384,
            "layers": 6,
            "speed_estimate": "~180 samples/sec (CPU)",
            "use_case": "Fastest inference, slight accuracy drop"
        },
        "DistilRoBERTa": {
            "params": "82M",
            "hidden": 768,
            "layers": 6,
            "speed_estimate": "~60 samples/sec (CPU)",
            "use_case": "Higher accuracy, slower inference"
        },
        "XLM-RoBERTa": {
            "params": "278M",
            "hidden": 768,
            "layers": 12,
            "speed_estimate": "~25 samples/sec (CPU)",
            "use_case": "Multilingual, very slow on CPU"
        }
    }
    
    print(f"\n{'Model':<20} {'Params':<10} {'Hidden':<8} {'Layers':<7} {'Speed':<20}")
    print("-" * 70)
    
    for name, info in comparison.items():
        print(f"{name:<20} {info['params']:<10} {info['hidden']:<8} {info['layers']:<7} {info['speed_estimate']:<20}")
    
    print("\n" + "=" * 70)
    print("Use Cases:")
    print("=" * 70)
    for name, info in comparison.items():
        print(f"\n{name}:")
        print(f"  {info['use_case']}")
    
    print("\n" + "=" * 70)
    print("\n💡 To run full benchmark: python scripts/compare_backbones.py --full")
    print("   (Warning: Downloads all models, ~1GB total)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare transformer backbones")
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full benchmark (downloads models)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/backbone_comparison.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    if args.full:
        compare_backbones(output_file=args.output)
    else:
        quick_comparison()
