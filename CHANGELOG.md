# Changelog

All notable changes to Project Aurora Echo will be documented in this file.

## [2.0.0] - 2026-02-16

### Added
- **NVIDIA DCGM Integration**: Real-time GPU monitoring with Prometheus/Grafana dashboards.
- **TensorRT Optimization**: 2-5x faster ASR inference with TensorRT-optimized Whisper models.
- **Triton Inference Server**: Production-grade model serving with vLLM backend for LLM.
- **NVIDIA NIM Microservices**: Cloud-optimized inference microservices for LLM and ASR.
- **Enhanced Observability**: Advanced Grafana dashboards correlating GPU and application metrics.
- **Multi-GPU Support**: Configuration for scaling across multiple NVIDIA GPUs.
- **Performance Benchmarking**: Scripts to compare TensorRT vs faster-whisper performance.

### Changed
- **Updated ASR Service**: Added TensorRT backend with automatic fallback to faster-whisper.
- **Updated LLM Service**: Added NVIDIA NIM provider and improved provider ordering.
- **Enhanced Docker Configuration**: Added DCGM exporter, Triton, and NIM services.
- **Updated Documentation**: Comprehensive README files for each NVIDIA technology.

### Fixed
- Improved error handling and retry logic for inference providers.
- Enhanced observability metrics for better monitoring.

## [1.0.0] - Initial Release

- Basic meeting copilot with faster-whisper ASR and vLLM LLM.
- Real-time audio streaming via WebSocket.
- Speaker diarization with pyannote.audio.
- Basic observability with Prometheus metrics.
