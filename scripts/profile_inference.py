"""Performance profiling script for Miyraa NLP inference pipeline.

Benchmarks:
1. PyTorch model inference (CPU)
2. ONNX model inference (CPU)
3. ONNX quantized model inference
4. PII scrubbing overhead
5. End-to-end API latency
6. Throughput (requests per second)
7. Memory usage

Usage:
    python scripts/profile_inference.py --samples 100 --warmup 10
"""

import argparse
import time
import json
import psutil
import os
from pathlib import Path
from typing import List, Dict
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# Try to import torch and onnxruntime
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from src.nlp.safety.pii_scrub import scrub_pii


class PerformanceProfiler:
    """Profile inference performance across different backends."""
    
    def __init__(self, samples: int = 100, warmup: int = 10):
        self.samples = samples
        self.warmup = warmup
        self.results = {}
        
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def generate_test_texts(self, n: int) -> List[str]:
        """Generate test texts of varying lengths."""
        base_texts = [
            "I'm feeling great today!",
            "This is absolutely terrible and disappointing.",
            "The weather is nice and calm, perfect for a walk in the park.",
            "I can't believe how amazing this experience has been so far!",
            "Contact me at john.doe@example.com or call +1-555-123-4567 for more details.",
        ]
        texts = []
        for i in range(n):
            text = base_texts[i % len(base_texts)]
            # Vary text length
            if i % 3 == 0:
                text = text + " " + text
            texts.append(text)
        return texts
    
    def benchmark_function(self, func, inputs: List, name: str) -> Dict:
        """Benchmark a function with warmup and multiple runs."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")
        
        # Warmup
        print(f"Warmup ({self.warmup} iterations)...")
        for inp in inputs[:self.warmup]:
            _ = func(inp)
        
        # Actual benchmark
        print(f"Running benchmark ({self.samples} iterations)...")
        latencies = []
        mem_before = self.get_memory_usage()
        
        for inp in inputs[:self.samples]:
            start = time.perf_counter()
            _ = func(inp)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        mem_after = self.get_memory_usage()
        
        results = {
            "name": name,
            "samples": self.samples,
            "latency_ms": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "std": np.std(latencies),
            },
            "throughput_rps": 1000 / np.mean(latencies),
            "memory_delta_mb": mem_after - mem_before,
            "memory_final_mb": mem_after,
        }
        
        self.print_results(results)
        return results
    
    def print_results(self, results: Dict):
        """Pretty print benchmark results."""
        print(f"\nResults for: {results['name']}")
        print(f"  Samples: {results['samples']}")
        print(f"  Mean latency: {results['latency_ms']['mean']:.2f} ms")
        print(f"  Median latency: {results['latency_ms']['median']:.2f} ms")
        print(f"  P95 latency: {results['latency_ms']['p95']:.2f} ms")
        print(f"  P99 latency: {results['latency_ms']['p99']:.2f} ms")
        print(f"  Throughput: {results['throughput_rps']:.2f} req/s")
        print(f"  Memory delta: {results['memory_delta_mb']:.2f} MB")
    
    def benchmark_pii_scrubbing(self, texts: List[str]):
        """Benchmark PII scrubbing performance."""
        def scrub_func(text):
            return scrub_pii(text, use_presidio=False)
        
        results = self.benchmark_function(scrub_func, texts, "PII Scrubbing (Regex)")
        self.results["pii_regex"] = results
        
        # Try Presidio if available
        try:
            def scrub_presidio_func(text):
                return scrub_pii(text, use_presidio=True)
            
            results_presidio = self.benchmark_function(
                scrub_presidio_func, texts, "PII Scrubbing (Presidio)"
            )
            self.results["pii_presidio"] = results_presidio
        except Exception as e:
            print(f"Presidio benchmarking skipped: {e}")
    
    def benchmark_torch_inference(self, texts: List[str], model_path: str):
        """Benchmark PyTorch model inference."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping...")
            return
        
        try:
            print(f"Loading PyTorch model from {model_path}...")
            # This is a placeholder - actual model loading depends on your architecture
            checkpoint = torch.load(model_path, map_location="cpu")
            print("PyTorch model loaded!")
            
            def infer_func(text):
                # Placeholder for actual inference
                # In real scenario, tokenize + forward pass
                return {"dummy": True}
            
            results = self.benchmark_function(infer_func, texts, "PyTorch Inference (CPU)")
            self.results["pytorch_cpu"] = results
        except Exception as e:
            print(f"PyTorch benchmarking failed: {e}")
    
    def benchmark_onnx_inference(self, texts: List[str], onnx_path: str):
        """Benchmark ONNX model inference."""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available, skipping...")
            return
        
        try:
            print(f"Loading ONNX model from {onnx_path}...")
            session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']
            )
            print("ONNX model loaded!")
            
            def infer_func(text):
                # Placeholder for actual inference
                # In real scenario: tokenize + session.run()
                return {"dummy": True}
            
            results = self.benchmark_function(infer_func, texts, f"ONNX Inference ({Path(onnx_path).name})")
            self.results[f"onnx_{Path(onnx_path).stem}"] = results
        except Exception as e:
            print(f"ONNX benchmarking failed: {e}")
    
    def generate_report(self, output_path: str):
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "torch_available": TORCH_AVAILABLE,
                "onnx_available": ONNX_AVAILABLE,
            },
            "benchmark_config": {
                "samples": self.samples,
                "warmup": self.warmup,
            },
            "results": self.results,
        }
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Report saved to: {output_path}")
        print(f"{'='*60}")
        
        # Print comparison
        self.print_comparison()
    
    def print_comparison(self):
        """Print comparison table of all benchmarks."""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Benchmark':<35} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput (req/s)'}")
        print("-"*80)
        
        for key, result in self.results.items():
            name = result['name'][:34]
            mean = result['latency_ms']['mean']
            p95 = result['latency_ms']['p95']
            throughput = result['throughput_rps']
            print(f"{name:<35} {mean:<12.2f} {p95:<12.2f} {throughput:.2f}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Profile Miyraa inference performance")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--torch-model", type=str, default="outputs/production_checkpoint.pt",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx-model", type=str, default="outputs/xlm-roberta.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--onnx-quant", type=str, default="outputs/xlm-roberta.quant.onnx",
                        help="Path to quantized ONNX model")
    parser.add_argument("--output", type=str, default="reports/performance_profile.json",
                        help="Output path for performance report")
    
    args = parser.parse_args()
    
    profiler = PerformanceProfiler(samples=args.samples, warmup=args.warmup)
    
    # Generate test texts
    texts = profiler.generate_test_texts(args.samples + args.warmup)
    
    # Run benchmarks
    print("\n" + "="*80)
    print("MIYRAA PERFORMANCE PROFILING")
    print("="*80)
    
    # 1. PII Scrubbing
    profiler.benchmark_pii_scrubbing(texts)
    
    # 2. PyTorch inference
    if Path(args.torch_model).exists():
        profiler.benchmark_torch_inference(texts, args.torch_model)
    else:
        print(f"\nPyTorch model not found at {args.torch_model}, skipping...")
    
    # 3. ONNX inference
    if Path(args.onnx_model).exists():
        profiler.benchmark_onnx_inference(texts, args.onnx_model)
    else:
        print(f"\nONNX model not found at {args.onnx_model}, skipping...")
    
    # 4. ONNX quantized inference
    if Path(args.onnx_quant).exists():
        profiler.benchmark_onnx_inference(texts, args.onnx_quant)
    else:
        print(f"\nQuantized ONNX model not found at {args.onnx_quant}, skipping...")
    
    # Generate report
    profiler.generate_report(args.output)
    
    print("\n✅ Performance profiling completed!")


if __name__ == "__main__":
    main()
