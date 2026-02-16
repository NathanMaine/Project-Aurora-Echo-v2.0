"""Prometheus metrics for the meeting assistant."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# --- Core Inference Metrics ---
INFERENCE_JOBS_TOTAL = Counter(
    "meeting_assistant_inference_jobs_total",
    "Total number of inference jobs processed",
)

INFERENCE_JOB_FAILURES = Counter(
    "meeting_assistant_inference_job_failures_total",
    "Total number of inference jobs that resulted in error",
)

# --- Latency Histograms (per service) ---
ASR_LATENCY = Histogram(
    "meeting_assistant_asr_latency_seconds",
    "Latency of ASR transcription per job",
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 40, float("inf")),
)

DIARIZATION_LATENCY = Histogram(
    "meeting_assistant_diarization_latency_seconds",
    "Latency of diarization per job",
    buckets=(0.25, 0.5, 1, 2, 5, 10, float("inf")),
)

LLM_LATENCY = Histogram(
    "meeting_assistant_llm_latency_seconds",
    "Latency of LLM summarization per job",
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, float("inf")),
)

INFERENCE_JOB_DURATION = Histogram(
    "meeting_assistant_inference_job_duration_seconds",
    "End-to-end latency for inference jobs",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, float("inf")),
)

# --- Per-Provider Metrics ---
ASR_PROVIDER_LATENCY = Histogram(
    "meeting_assistant_asr_provider_latency_seconds",
    "Latency of ASR transcription per provider",
    labelnames=("provider",),
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 40, float("inf")),
)

ASR_PROVIDER_ERRORS = Counter(
    "meeting_assistant_asr_provider_errors_total",
    "Total number of ASR errors per provider",
    labelnames=("provider",),
)

LLM_PROVIDER_LATENCY = Histogram(
    "meeting_assistant_llm_provider_latency_seconds",
    "Latency of LLM summarization per provider",
    labelnames=("provider",),
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, float("inf")),
)

LLM_PROVIDER_ERRORS = Counter(
    "meeting_assistant_llm_provider_errors_total",
    "Total number of LLM errors per provider",
    labelnames=("provider",),
)

# --- TensorRT Specific Metrics ---
TENSORRT_ENGINE_LOAD_TIME = Histogram(
    "meeting_assistant_tensorrt_engine_load_time_seconds",
    "Time taken to load TensorRT engine",
    buckets=(0.1, 0.5, 1, 2, 5, 10, float("inf")),
)

TENSORRT_INFERENCE_LATENCY = Histogram(
    "meeting_assistant_tensorrt_inference_latency_seconds",
    "Latency of TensorRT inference per job",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, float("inf")),
)

TENSORRT_FALLBACKS = Counter(
    "meeting_assistant_tensorrt_fallbacks_total",
    "Total number of fallbacks to faster-whisper",
)

# --- GPU Correlation Metrics ---
GPU_UTILIZATION_CORRELATION = Gauge(
    "meeting_assistant_gpu_utilization_correlation",
    "Correlation between GPU utilization and inference latency",
    labelnames=("service",),
)

GPU_MEMORY_USAGE = Gauge(
    "meeting_assistant_gpu_memory_usage_bytes",
    "GPU memory usage per service",
    labelnames=("service",),
)

# --- Queue Metrics ---
QUEUE_DEPTH = Gauge(
    "meeting_assistant_inference_queue_depth",
    "Current depth of the inference queue",
    labelnames=("backend",),
)

QUEUE_WAIT_TIME = Histogram(
    "meeting_assistant_queue_wait_time_seconds",
    "Time jobs spend waiting in queue",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, float("inf")),
)

# --- Throughput Metrics ---
THROUGHPUT = Gauge(
    "meeting_assistant_throughput_jobs_per_second",
    "Current throughput in jobs per second",
    labelnames=("service",),
)

# --- Error Rate Metrics ---
ERROR_RATE = Gauge(
    "meeting_assistant_error_rate_percent",
    "Error rate percentage per service",
    labelnames=("service",),
)

# --- Helper Functions ---
def update_queue_depth(depth: int, backend: str) -> None:
    QUEUE_DEPTH.labels(backend=backend).set(depth)

def record_asr_provider_latency(latency: float, provider: str) -> None:
    ASR_PROVIDER_LATENCY.labels(provider=provider).observe(latency)

def record_asr_provider_error(provider: str) -> None:
    ASR_PROVIDER_ERRORS.labels(provider=provider).inc()

def record_llm_provider_latency(latency: float, provider: str) -> None:
    LLM_PROVIDER_LATENCY.labels(provider=provider).observe(latency)

def record_llm_provider_error(provider: str) -> None:
    LLM_PROVIDER_ERRORS.labels(provider=provider).inc()

def record_tensorrt_engine_load_time(load_time: float) -> None:
    TENSORRT_ENGINE_LOAD_TIME.observe(load_time)

def record_tensorrt_inference_latency(latency: float) -> None:
    TENSORRT_INFERENCE_LATENCY.observe(latency)

def record_tensorrt_fallback() -> None:
    TENSORRT_FALLBACKS.inc()

def update_gpu_correlation(correlation: float, service: str) -> None:
    GPU_UTILIZATION_CORRELATION.labels(service=service).set(correlation)

def update_gpu_memory_usage(bytes_used: int, service: str) -> None:
    GPU_MEMORY_USAGE.labels(service=service).set(bytes_used)

def update_throughput(jobs_per_second: float, service: str) -> None:
    THROUGHPUT.labels(service=service).set(jobs_per_second)

def update_error_rate(rate_percent: float, service: str) -> None:
    ERROR_RATE.labels(service=service).set(rate_percent)

def record_queue_wait_time(wait_time: float) -> None:
    QUEUE_WAIT_TIME.observe(wait_time)
