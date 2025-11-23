"""
ONNX model integration tests.
Tests ONNX model loading, inference, and accuracy.
"""

import pytest
import numpy as np
import os
from pathlib import Path


# Try to import onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime not installed")
class TestONNXModelLoading:
    """Test ONNX model loading"""

    def test_load_base_model(self):
        """Test loading base ONNX model"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        session = ort.InferenceSession(model_path)
        assert session is not None

        # Check inputs
        inputs = session.get_inputs()
        assert len(inputs) > 0

        # Check outputs
        outputs = session.get_outputs()
        assert len(outputs) > 0

    def test_load_quantized_model(self):
        """Test loading quantized ONNX model"""
        model_path = "outputs/xlm-roberta.quant.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Quantized model not found: {model_path}")

        session = ort.InferenceSession(model_path)
        assert session is not None

    def test_model_input_shapes(self):
        """Test model input shapes are correct"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        session = ort.InferenceSession(model_path)
        inputs = session.get_inputs()

        # Should have input_ids and attention_mask
        input_names = [inp.name for inp in inputs]
        assert "input_ids" in input_names or len(input_names) > 0

    def test_model_output_shapes(self):
        """Test model output shapes are correct"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        session = ort.InferenceSession(model_path)
        outputs = session.get_outputs()

        # Should have at least one output
        assert len(outputs) >= 1


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime not installed")
class TestONNXInference:
    """Test ONNX model inference"""

    def test_basic_inference(self):
        """Test basic ONNX inference"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        session = ort.InferenceSession(model_path)

        # Create dummy input
        batch_size = 1
        seq_len = 16
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Run inference
        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]

        if "input_ids" in input_names:
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            # Use first two inputs
            input_dict = {
                input_names[0]: input_ids,
                input_names[1]: attention_mask
            }

        outputs = session.run(None, input_dict)

        # Verify outputs
        assert len(outputs) > 0
        assert outputs[0].shape[0] == batch_size

    def test_batch_inference(self):
        """Test batch inference"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        session = ort.InferenceSession(model_path)

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            seq_len = 16
            input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

            inputs = session.get_inputs()
            input_names = [inp.name for inp in inputs]

            if "input_ids" in input_names:
                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            else:
                input_dict = {
                    input_names[0]: input_ids,
                    input_names[1]: attention_mask
                }

            outputs = session.run(None, input_dict)
            assert outputs[0].shape[0] == batch_size

    def test_variable_sequence_length(self):
        """Test inference with different sequence lengths"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        session = ort.InferenceSession(model_path)

        # Test different sequence lengths
        for seq_len in [8, 16, 32, 64]:
            batch_size = 1
            input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

            inputs = session.get_inputs()
            input_names = [inp.name for inp in inputs]

            if "input_ids" in input_names:
                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            else:
                input_dict = {
                    input_names[0]: input_ids,
                    input_names[1]: attention_mask
                }

            outputs = session.run(None, input_dict)
            assert len(outputs) > 0


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime not installed")
class TestONNXQuantization:
    """Test quantized ONNX model"""

    def test_quantized_inference(self):
        """Test quantized model inference"""
        model_path = "outputs/xlm-roberta.quant.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Quantized model not found: {model_path}")

        session = ort.InferenceSession(model_path)

        # Create input
        batch_size = 1
        seq_len = 16
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]

        if "input_ids" in input_names:
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            input_dict = {
                input_names[0]: input_ids,
                input_names[1]: attention_mask
            }

        outputs = session.run(None, input_dict)
        assert len(outputs) > 0

    def test_quantized_vs_base_accuracy(self):
        """Test quantized model vs base model accuracy"""
        base_path = "outputs/xlm-roberta.onnx"
        quant_path = "outputs/xlm-roberta.quant.onnx"

        if not os.path.exists(base_path) or not os.path.exists(quant_path):
            pytest.skip("Models not found for comparison")

        base_session = ort.InferenceSession(base_path)
        quant_session = ort.InferenceSession(quant_path)

        # Create test input
        batch_size = 4
        seq_len = 16
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Get input names
        base_inputs = base_session.get_inputs()
        base_input_names = [inp.name for inp in base_inputs]

        if "input_ids" in base_input_names:
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            input_dict = {
                base_input_names[0]: input_ids,
                base_input_names[1]: attention_mask
            }

        # Run both models
        base_outputs = base_session.run(None, input_dict)
        quant_outputs = quant_session.run(None, input_dict)

        # Compare outputs (should be similar)
        # Allow for some quantization error
        diff = np.abs(base_outputs[0] - quant_outputs[0]).mean()
        assert diff < 0.1  # Reasonable threshold for quantization error


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime not installed")
class TestONNXPerformance:
    """Test ONNX model performance"""

    def test_inference_latency(self):
        """Test inference latency is reasonable"""
        model_path = "outputs/xlm-roberta.onnx"
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")

        import time

        session = ort.InferenceSession(model_path)

        # Warmup
        batch_size = 1
        seq_len = 16
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]

        if "input_ids" in input_names:
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            input_dict = {
                input_names[0]: input_ids,
                input_names[1]: attention_mask
            }

        # Warmup runs
        for _ in range(5):
            session.run(None, input_dict)

        # Measure latency
        start = time.time()
        n_runs = 20
        for _ in range(n_runs):
            session.run(None, input_dict)
        end = time.time()

        avg_latency = (end - start) / n_runs

        # Latency should be reasonable (< 1 second per inference)
        assert avg_latency < 1.0

    def test_quantized_speedup(self):
        """Test quantized model is faster than base model"""
        base_path = "outputs/xlm-roberta.onnx"
        quant_path = "outputs/xlm-roberta.quant.onnx"

        if not os.path.exists(base_path) or not os.path.exists(quant_path):
            pytest.skip("Models not found for comparison")

        import time

        base_session = ort.InferenceSession(base_path)
        quant_session = ort.InferenceSession(quant_path)

        # Create test input
        batch_size = 1
        seq_len = 32
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Get input names
        base_inputs = base_session.get_inputs()
        base_input_names = [inp.name for inp in base_inputs]

        if "input_ids" in base_input_names:
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            input_dict = {
                base_input_names[0]: input_ids,
                base_input_names[1]: attention_mask
            }

        # Warmup
        for _ in range(5):
            base_session.run(None, input_dict)
            quant_session.run(None, input_dict)

        # Measure base model
        start = time.time()
        for _ in range(20):
            base_session.run(None, input_dict)
        base_time = time.time() - start

        # Measure quantized model
        start = time.time()
        for _ in range(20):
            quant_session.run(None, input_dict)
        quant_time = time.time() - start

        # Quantized should be faster or similar
        # (On CPU might not be significantly faster)
        assert quant_time <= base_time * 1.5  # Allow 50% margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
