#!/usr/bin/env python3
"""Benchmark TensorRT vs faster-whisper performance."""

import time
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_tensorrt():
    """Benchmark TensorRT Whisper pipeline."""
    try:
        from services.tensorrt_whisper import TensorRTWhisperModel
        
        # Check if TensorRT engine exists
        engine_path = "./models/whisper-medium-fp16.plan"
        if not Path(engine_path).exists():
            logger.warning(f"TensorRT engine not found at {engine_path}")
            logger.warning("Run: python convert_to_tensorrt.py --model-size medium")
            return None
        
        # Initialize TensorRT model
        logger.info("Initializing TensorRT Whisper model...")
        start_time = time.time()
        model = TensorRTWhisperModel(
            model_size="medium",
            engine_path=engine_path,
            device="cuda",
            compute_type="fp16",
        )
        init_time = time.time() - start_time
        logger.info(f"TensorRT model initialization: {init_time:.2f}s")
        
        # Generate dummy audio
        sample_rate = 16000
        duration = 5  # seconds
        audio = np.random.randn(sample_rate * duration).astype(np.float32)
        
        # Warmup
        logger.info("Running warmup inference...")
        for _ in range(3):
            segments, info = model.transcribe(audio, sample_rate=sample_rate)
        
        # Benchmark inference
        logger.info("Benchmarking inference...")
        n_runs = 10
        inference_times = []
        
        for i in range(n_runs):
            start_time = time.time()
            segments, info = model.transcribe(audio, sample_rate=sample_rate)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            logger.info(f"Run {i+1}: {inference_time:.3f}s")
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        logger.info(f"TensorRT Inference - Avg: {avg_time:.3f}s, Std: {std_time:.3f}s")
        
        return {
            "backend": "tensorrt",
            "init_time": init_time,
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "sample_duration": duration,
        }
    
    except ImportError as e:
        logger.error(f"TensorRT not available: {e}")
        return None
    except Exception as e:
        logger.error(f"TensorRT benchmark failed: {e}")
        return None


def benchmark_faster_whisper():
    """Benchmark faster-whisper performance."""
    try:
        from faster_whisper import WhisperModel
        
        # Initialize faster-whisper model
        logger.info("Initializing faster-whisper model...")
        start_time = time.time()
        model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16",
        )
        init_time = time.time() - start_time
        logger.info(f"faster-whisper model initialization: {init_time:.2f}s")
        
        # Generate dummy audio
        sample_rate = 16000
        duration = 5  # seconds
        audio = np.random.randn(sample_rate * duration).astype(np.float32)
        
        # Warmup
        logger.info("Running warmup inference...")
        for _ in range(3):
            segments, info = model.transcribe(audio, sample_rate=sample_rate)
        
        # Benchmark inference
        logger.info("Benchmarking inference...")
        n_runs = 10
        inference_times = []
        
        for i in range(n_runs):
            start_time = time.time()
            segments, info = model.transcribe(audio, sample_rate=sample_rate)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            logger.info(f"Run {i+1}: {inference_time:.3f}s")
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        logger.info(f"faster-whisper Inference - Avg: {avg_time:.3f}s, Std: {std_time:.3f}s")
        
        return {
            "backend": "faster-whisper",
            "init_time": init_time,
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "sample_duration": duration,
        }
    
    except ImportError as e:
        logger.error(f"faster-whisper not available: {e}")
        return None
    except Exception as e:
        logger.error(f"faster-whisper benchmark failed: {e}")
        return None


def main():
    """Run benchmarks and compare results."""
    logger.info("=== TensorRT vs faster-whisper Benchmark ===")
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Benchmark requires GPU.")
        return
    
    # Run benchmarks
    tensorrt_results = benchmark_tensorrt()
    faster_results = benchmark_faster_whisper()
    
    # Compare results
    logger.info("\n=== Comparison ===")
    if tensorrt_results and faster_results:
        speedup = faster_results["avg_inference_time"] / tensorrt_results["avg_inference_time"]
        logger.info(f"Speedup (TensorRT / faster-whisper): {speedup:.2f}x")
        
        print("\nSummary:")
        print(f"TensorRT - Init: {tensorrt_results['init_time']:.2f}s, Inference: {tensorrt_results['avg_inference_time']:.3f}s")
        print(f"faster-whisper - Init: {faster_results['init_time']:.2f}s, Inference: {faster_results['avg_inference_time']:.3f}s")
        print(f"Performance improvement: {speedup:.2f}x")
    elif tensorrt_results:
        print("\nOnly TensorRT benchmark completed:")
        print(f"TensorRT - Init: {tensorrt_results['init_time']:.2f}s, Inference: {tensorrt_results['avg_inference_time']:.3f}s")
    elif faster_results:
        print("\nOnly faster-whisper benchmark completed:")
        print(f"faster-whisper - Init: {faster_results['init_time']:.2f}s, Inference: {faster_results['avg_inference_time']:.3f}s")
    else:
        print("\nNo benchmarks completed successfully.")


if __name__ == "__main__":
    main()
