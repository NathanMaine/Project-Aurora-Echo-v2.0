# NVIDIA Triton Inference Server Deployment

## Overview
This integration replaces the standalone vLLM container with NVIDIA Triton Inference Server using the vLLM backend. This provides production-grade serving features: dynamic batching, model management, metrics, and multi-framework support.

## Changes Made
1. **docker-compose.yml**: Added `triton` service using `nvcr.io/nvidia/tritonserver:24.09-py3` image.
2. **Model Repository**: Created `models/triton/llama/1/config.pbtxt` for vLLM backend configuration.
3. **Environment Variables**: Updated `.env.docker` to point to Triton endpoint.
4. **LLM Service**: Updated `LLMService` to use Triton via existing VLLMProvider (OpenAI-compatible API).

## Prerequisites
- NVIDIA GPU with drivers and NVIDIA Container Toolkit installed.
- Docker and Docker Compose.
- Sufficient GPU memory for the model (approx 16GB for Llama-3-8B).

## Deployment
1. Ensure you are in the project root directory.
2. Start the stack:
   ```bash
   docker-compose up -d
   ```
3. Verify Triton is running:
   ```bash
   docker logs meeting-assistant-triton
   ```
   Look for "Server is ready" message.

## Model Loading
Triton is configured with `--model-control-mode=explicit` and `--load-model=llama`. The model will be loaded on startup.

To load/unload models manually:
```bash
# Load model
curl -X POST http://localhost:8000/v2/repository/models/llama/load

# Unload model
curl -X POST http://localhost:8000/v2/repository/models/llama/unload
```

## Endpoints
- **HTTP**: `http://localhost:8000`
- **gRPC**: `localhost:8001`
- **Metrics**: `localhost:8002`

### OpenAI-Compatible API
The vLLM backend exposes an OpenAI-compatible endpoint at `/v1/chat/completions`. The LLM service is configured to use:
- Base URL: `http://triton:8000/v1`
- Model ID: `llama`

## Configuration Details
### Model Configuration (`config.pbtxt`)
- Backend: `vllm`
- Model ID: `meta-llama-3-8b-instruct`
- Max batch size: 8
- Parameters: max_tokens=1024, temperature=0.7, top_p=0.9

### Docker Service
- GPU passthrough via `deploy.reservations.devices`
- Model repository mounted at `/models`
- Ports exposed: 8000 (HTTP), 8001 (gRPC), 8002 (metrics)

## Verification
1. Check Triton health:
   ```bash
   curl http://localhost:8000/v2/health/ready
   ```
2. List loaded models:
   ```bash
   curl http://localhost:8000/v2/repository/index
   ```
3. Test inference:
   ```bash
   curl -X POST http://localhost:8000/v2/models/llama/generate \
     -H "Content-Type: application/json" \
     -d '{"inputs": [{"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Hello, how are you?"]}]}'
   ```

## Monitoring
- Triton metrics are exposed at `http://localhost:8002/metrics` (Prometheus format).
- Integrate with existing Prometheus/Grafana stack by adding a scrape job.
- GPU metrics are available via DCGM integration.

## Fallback Strategy
If Triton fails to load, you can revert to the original vLLM container by:
1. Uncomment the `vllm` service in `docker-compose.yml`.
2. Comment the `triton` service.
3. Update `.env.docker` to point to `http://vllm:8001`.

## Performance Expectations
- **Throughput**: Dynamic batching improves throughput under concurrent loads.
- **Latency**: Comparable to standalone vLLM with potential improvements from optimized scheduling.
- **Resource Utilization**: Better GPU utilization via Triton's concurrent model execution.

## Next Steps
- Add multiple models (different sizes, languages).
- Enable model ensemble for complex pipelines.
- Implement Triton's rate limiting and request prioritization.
- Integrate Triton metrics with Prometheus for comprehensive observability.

## Troubleshooting
- **Model fails to load**: Check GPU memory availability. Reduce `max_batch_size` if needed.
- **HTTP 503**: Model not loaded. Verify model repository path and permissions.
- **Performance issues**: Monitor GPU metrics via DCGM dashboard.
- **API compatibility**: Ensure vLLM backend version supports OpenAI API format.

## References
- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [vLLM Backend for Triton](https://github.com/triton-inference-server/vllm_backend)
- [OpenAI-Compatible API](https://github.com/vllm-project/vllm/blob/main/docs/serving/openai_compatible_server.md)
