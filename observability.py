"""Prometheus metrics for the meeting assistant."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INFERENCE_JOBS_TOTAL = Counter(
    "meeting_assistant_inference_jobs_total",
    "Total number of inference jobs processed",
)

INFERENCE_JOB_FAILURES = Counter(
    "meeting_assistant_inference_job_failures_total",
    "Total number of inference jobs that resulted in error",
)

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

QUEUE_DEPTH = Gauge(
    "meeting_assistant_inference_queue_depth",
    "Current depth of the inference queue",
    labelnames=("backend",),
)


def update_queue_depth(depth: int, backend: str) -> None:
    QUEUE_DEPTH.labels(backend=backend).set(depth)
