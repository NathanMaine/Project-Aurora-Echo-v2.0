#!/usr/bin/env python3
"""Convert Whisper encoder to TensorRT engine.

This script converts the Whisper encoder (mel spectrogram to logits) to a TensorRT engine
using torch-tensorrt. The resulting engine can be loaded by TensorRTWhisperModel for
accelerated inference.

Usage:
    python convert_to_tensorrt.py --model-size medium --output-dir ./models
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    import torch
    import torch_tensorrt
    import whisper
    TENSORRT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Missing dependencies: {e}")
    TENSORRT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_whisper_encoder_to_tensorrt(
    model_size: str = "medium",
    output_dir: str = "./models",
    precision: str = "fp16",
    max_batch_size: int = 1,
) -> str:
    """Convert Whisper encoder to TensorRT engine.
    
    Args:
        model_size: Whisper model size (tiny, base, small, medium, large).
        output_dir: Directory to save TensorRT engine.
        precision: Precision mode (fp32, fp16, int8).
        max_batch_size: Maximum batch size for engine.
    
    Returns:
        Path to saved TensorRT engine file.
    """
    if not TENSORRT_AVAILABLE:
        raise RuntimeError(
            "TensorRT dependencies not installed. Install with: "
            "pip install torch-tensorrt tensorrt"
        )
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TensorRT conversion requires GPU.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    engine_filename = f"whisper-{model_size}-encoder-{precision}.plan"
    engine_path = output_path / engine_filename
    
    if engine_path.exists():
        logger.info(f"TensorRT engine already exists at {engine_path}")
        return str(engine_path)
    
    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size).cuda()
    model.eval()
    
    # Extract encoder for TensorRT conversion
    # Whisper encoder expects mel spectrogram of shape (batch, n_mels, n_frames)
    # where n_mels=80, n_frames=3000 for 30-second audio
    n_mels = 80
    n_frames = 3000
    
    # Create example input for tracing
    example_input = torch.randn(
        max_batch_size, n_mels, n_frames, device="cuda", dtype=torch.float32
    )
    
    # Create wrapper for encoder only
    class WhisperEncoderWrapper(torch.nn.Module):
        def __init__(self, whisper_model):
            super().__init__()
            self.encoder = whisper_model.encoder
            
        def forward(self, mel_spec):
            # Whisper encoder expects shape (batch, n_mels, n_frames)
            # Returns encoder output of shape (batch, n_audio_ctx, n_audio_state)
            return self.encoder(mel_spec)
    
    encoder_wrapper = WhisperEncoderWrapper(model).cuda()
    encoder_wrapper.eval()
    
    logger.info("Tracing encoder with TorchScript...")
    traced_encoder = torch.jit.trace(encoder_wrapper, example_input)
    
    logger.info(f"Compiling to TensorRT with precision {precision}...")
    # Configure TensorRT compilation settings
    trt_precision = torch_tensorrt.dtype.float16 if precision == "fp16" else torch_tensorrt.dtype.float32
    
    compile_settings = {
        "inputs": [torch_tensorrt.Input(
            shape=example_input.shape,
            dtype=trt_precision,
        )],
        "enabled_precisions": {trt_precision},
        "max_batch_size": max_batch_size,
        "workspace_size": 1 << 30,  # 1 GB
        "truncate_long_and_double": True,
    }
    
    trt_encoder = torch_tensorrt.compile(traced_encoder, **compile_settings)
    
    logger.info(f"Saving TensorRT engine to {engine_path}")
    torch.jit.save(trt_encoder, str(engine_path))
    
    # Save metadata about the conversion
    metadata_path = output_path / f"whisper-{model_size}-encoder-metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"model_size: {model_size}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"input_shape: {example_input.shape}\n")
        f.write(f"max_batch_size: {max_batch_size}\n")
    
    logger.info("Conversion complete!")
    logger.info(f"Engine saved to: {engine_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return str(engine_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Whisper encoder to TensorRT engine.")
    parser.add_argument(
        "--model-size",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save TensorRT engine",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for engine",
    )
    
    args = parser.parse_args()
    
    try:
        engine_path = convert_whisper_encoder_to_tensorrt(
            model_size=args.model_size,
            output_dir=args.output_dir,
            precision=args.precision,
            max_batch_size=args.max_batch_size,
        )
        print(f"TensorRT engine saved to: {engine_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
