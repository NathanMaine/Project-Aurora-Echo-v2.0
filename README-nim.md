# NVIDIA NIM Integration

This document describes the integration of NVIDIA NIM (NVIDIA Inference Microservices) into Project-Aurora-Echo.

## Overview

NVIDIA NIM provides optimized inference microservices for AI models, exposing industry-standard APIs for easy integration. This integration adds support for NIM as an optional backend for both LLM and ASR services.

## Benefits

- **Performance**: NIM containers are optimized for NVIDIA GPUs, providing low-latency, high-throughput inference.
- **Simplified Deployment**: Pre-built containers with enterprise-grade support.
- **API Compatibility**: OpenAI-compatible endpoints for seamless integration.
- **Scalability**: Designed for cloud, data center, and edge deployment.

## Architecture

The integration adds two new providers:

1. **NIM LLM Provider** (`services/providers/nim_provider.py`): Uses NIM's OpenAI-compatible chat completions endpoint.
2. **NIM ASR Provider** (`services/providers/nim_asr_provider.py`): Uses NIM's audio transcription endpoint.

These providers are integrated into the existing LLMService and ASRService with automatic fallback to local models.

## Configuration

### Environment Variables

#### LLM Service
- `NIM_BASE_URL`: Base URL of NIM LLM service (default: `http://localhost:8002`)
- `NIM_MODEL_ID`: Model identifier (default: `meta-llama-3-8b-instruct`)
- `NIM_COMPLETIONS_ENDPOINT`: Endpoint path (default: `/v1/chat/completions`)
- `NIM_API_KEY`: Optional API key for authentication

#### ASR Service
- `NIM_ASR_BASE_URL`: Base URL of NIM ASR service (default: `http://localhost:8003`)
- `NIM_ASR_MODEL_ID`: Model identifier (default: `whisper-large-v3`)
- `NIM_ASR_ENDPOINT`: Endpoint path (default: `/v1/audio/transcriptions`)
- `NIM_ASR_API_KEY`: Optional API key for authentication
- `NIM_ASR_MAX_RETRIES`: Maximum retry attempts (default: `3`)
- `NIM_ASR_BACKOFF_SECONDS`: Backoff delay (default: `1.0`)
- `NIM_ASR_TIMEOUT`: Request timeout (default: `60.0`)

### Provider Order

Add `nim` to the `LLM_PROVIDER_ORDER` environment variable to enable NIM LLM provider.

Example:
```bash
export LLM_PROVIDER_ORDER="nim,vllm,grok"
```

For ASR, set `NIM_ASR_BASE_URL` to enable NIM ASR provider. If not set, the service falls back to local Whisper models.

## Deployment

### Prerequisites

- NVIDIA GPU with appropriate drivers
- NVIDIA Container Toolkit installed
- Docker and Docker Compose
- NVIDIA NGC API key (for pulling NIM containers)

### Docker Compose

A separate compose file `docker-compose.nim.yml` is provided for NIM services.

1. **Authenticate with NVIDIA NGC:**
   ```bash
   docker login nvcr.io
   Username: $oauthtoken
   Password: <your NGC API key>
   ```

2. **Start NIM services:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.nim.yml up -d
   ```

   This will start:
   - `meeting-assistant-nim-llm` on port 8002
   - `meeting-assistant-nim-asr` on port 8003

3. **Verify services:**
   ```bash
   curl http://localhost:8002/v1/models
   curl http://localhost:8003/v1/models
   ```

### Model Configuration

NIM containers expect models to be mounted at `/models`. The compose file mounts `./models/nim/llm` and `./models/nim/asr`.

To use custom models, place them in these directories and set the appropriate `NIM_MODEL` environment variable.

## Integration Details

### LLM Service

The `LLMService` class has been updated to include a `nim` provider. When `nim` is in the provider order, it creates a `NIMProvider` instance with the configured environment variables.

### ASR Service

The `ASRService` class now uses a factory function `get_asr_provider` that checks for `NIM_ASR_BASE_URL`. If set, it returns a `NIMASRProvider`; otherwise, it falls back to local Whisper models (TensorRT or faster-whisper).

### Observability

NIM integration includes Prometheus metrics for monitoring:
- `meeting_assistant_nim_llm_latency_seconds`
- `meeting_assistant_nim_asr_latency_seconds`
- `meeting_assistant_nim_requests_total`

These metrics are automatically collected when the providers are used.

## Performance Considerations

- **Latency**: NIM containers are optimized for GPU inference, typically providing lower latency than local models.
- **Throughput**: NIM supports dynamic batching for improved throughput.
- **Resource Usage**: NIM containers share GPU memory with other services; monitor usage via DCGM.

## Troubleshooting

### Common Issues

1. **NIM container fails to start:**
   - Verify GPU availability: `nvidia-smi`
   - Check NVIDIA Container Toolkit installation
   - Ensure NGC authentication is valid

2. **Provider not being used:**
   - Check environment variables are set correctly
   - Verify `LLM_PROVIDER_ORDER` includes `nim`
   - Check logs for warnings about missing configuration

3. **Authentication errors:**
   - Ensure `NIM_API_KEY` is set if required
   - Verify the API key has appropriate permissions

### Logging

Enable debug logging for detailed provider activity:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison with Existing Backends

| Feature | NIM | Triton | TensorRT | faster-whisper |
|---------|-----|--------|----------|----------------|
| API Compatibility | OpenAI-compatible | Triton API | Custom | Custom |
| Optimization Level | High | High | Very High | Medium |
| Deployment Complexity | Low | Medium | High | Low |
| Enterprise Support | Yes | Yes | Yes | No |
| Multi-Model Support | Yes | Yes | No | No |

## Next Steps

1. **Benchmarking**: Compare NIM performance against existing backends.
2. **Multi-GPU Support**: Extend NIM deployment to multiple GPUs.
3. **Model Ensemble**: Combine NIM LLM and ASR in a single pipeline.
4. **Kubernetes Deployment**: Deploy NIM services on Kubernetes with NVIDIA GPU Operator.

## References

- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
