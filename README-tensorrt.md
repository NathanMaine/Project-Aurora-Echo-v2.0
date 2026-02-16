# TensorRT Optimization for Whisper ASR

## Overview
This integration adds TensorRT acceleration to the Whisper ASR pipeline, providing 2-5x inference speedup compared to standard PyTorch/faster-whisper implementations.

## Architecture
```
Audio Input → Mel Spectrogram → TensorRT Engine → Logits → Token Decoder → Text Output
```

## Components
1. **TensorRT Conversion Script** (`convert_to_tensorrt.py`): Converts Whisper encoder to TensorRT engine.
2. **TensorRT Whisper Pipeline** (`services/tensorrt_whisper.py`): Full ASR pipeline with TensorRT acceleration.
3. **ASR Service Integration**: Automatic fallback to faster-whisper when TensorRT engine is unavailable.
4. **Benchmark Script** (`benchmark_tensorrt.py`): Performance comparison tool.

## Prerequisites
- NVIDIA GPU with CUDA 12.2+ and TensorRT 8.6+
- NVIDIA Container Toolkit for Docker deployment
- PyTorch with CUDA support
- TensorRT Python package (`pip install tensorrt torch-tensorrt`)

## Installation
1. Install TensorRT dependencies:
   ```bash
   pip install tensorrt torch-tensorrt
   ```

2. Update Dockerfile with TensorRT runtime:
   ```dockerfile
   RUN apt-get update && \
       apt-get install -y --no-install-recommends \
           libnvinfer9 \
           libnvinfer-plugin9
   ```

## Conversion Process
### Step 1: Convert Whisper Model to TensorRT Engine
```bash
python convert_to_tensorrt.py \
    --model-size medium \
    --precision fp16 \
    --max-batch-size 8 \
    --output-dir ./models
```

**Options:**
- `--model-size`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- `--precision`: Precision mode (`fp32`, `fp16`, `int8`)
- `--max-batch-size`: Maximum batch size for engine
- `--output-dir`: Directory to save TensorRT engine

### Step 2: Verify Conversion
```bash
# Check generated files
ls -la ./models/
# Should see:
# - whisper-medium-fp16.plan          (TensorRT engine)
# - whisper-medium-encoder-metadata.txt (Conversion metadata)
```

## Usage
### Automatic Fallback
The ASR service automatically detects and uses TensorRT engines:
```python
from services.asr_service import ASRService

# TensorRT will be used if engine exists and CUDA is available
asr = ASRService(model_size="medium", device="cuda")
```

### Manual Usage
```python
from services.tensorrt_whisper import TensorRTWhisperModel

# Initialize TensorRT pipeline
model = TensorRTWhisperModel(
    model_size="medium",
    engine_path="./models/whisper-medium-fp16.plan",
    device="cuda",
    compute_type="fp16",
)

# Transcribe audio
segments, info = model.transcribe(
    audio=audio_data,
    sample_rate=16000,
    language="en",
    beam_size=5,
)
```

## Performance Benchmarking
Run the benchmark script to compare TensorRT vs faster-whisper:
```bash
python benchmark_tensorrt.py
```

**Expected Results:**
- **Initialization**: TensorRT engines load faster than full Whisper models
- **Inference**: 2-5x speedup depending on batch size and precision
- **Memory**: Reduced GPU memory footprint with FP16/INT8 quantization

## Integration with Docker
### Updated Dockerfile
The `docker/api.Dockerfile` includes:
```dockerfile
# TensorRT runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libnvinfer9 \
        libnvinfer-plugin9

# TensorRT Python packages
RUN pip install tensorrt torch-tensorrt
```

### Environment Variables
No additional environment variables required. The system automatically detects TensorRT engines.

## Fallback Mechanism
If TensorRT engine is not available, the system falls back to faster-whisper:
1. Checks for TensorRT engine file in `./models/`
2. Validates CUDA availability
3. Falls back to faster-whisper if conditions not met
4. Logs warning message with fallback reason

## Monitoring
TensorRT inference metrics are exposed via:
1. **DCGM Integration**: GPU utilization, memory usage
2. **Prometheus**: Custom metrics for TensorRT inference latency
3. **Grafana Dashboard**: ASR performance monitoring

## Troubleshooting
### Common Issues
1. **Engine not found**: Run conversion script first
2. **CUDA unavailable**: Ensure GPU drivers and CUDA are installed
3. **TensorRT version mismatch**: Use TensorRT 8.6+ compatible with CUDA 12.2
4. **Memory errors**: Reduce batch size or use FP16/INT8 precision

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization
### Precision Selection
- **FP32**: Maximum accuracy, slower inference
- **FP16**: Balanced accuracy/speed (recommended)
- **INT8**: Maximum speed, quantization required

### Batch Size Tuning
Adjust `--max-batch-size` during conversion:
```bash
# For high-throughput scenarios
python convert_to_tensorrt.py --max-batch-size 16

# For low-latency scenarios
python convert_to_tensorrt.py --max-batch-size 1
```

### Dynamic Shapes
For variable-length audio, enable dynamic shapes in conversion script.

## Next Steps
1. **Quantization**: Add INT8 quantization support for further speedup
2. **Dynamic Batching**: Implement dynamic batching for variable-length inputs
3. **Multi-GPU**: Scale across multiple GPUs
4. **Streaming**: Real-time streaming inference with TensorRT

## References
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Whisper Model Architecture](https://github.com/openai/whisper)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
