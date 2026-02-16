# NVIDIA DCGM Integration for Project-Aurora-Echo

## Overview
This integration adds NVIDIA Data Center GPU Manager (DCGM) monitoring to the existing Prometheus/Grafana observability stack.

## Changes Made
1. **docker-compose.yml**: Added `dcgm-exporter` service with NVIDIA GPU passthrough.
2. **prometheus.yml**: Added scrape job for DCGM metrics on port 9400.
3. **Grafana Dashboard**: Created `dcgm-gpu-monitoring.json` with 6 panels for GPU metrics.

## Prerequisites
- Host system with NVIDIA GPU and drivers installed.
- NVIDIA Container Toolkit installed and configured for Docker.
- Docker and Docker Compose installed.

## Deployment
1. Ensure you are in the project root directory.
2. Start the stack:
   ```bash
   docker-compose up -d
   ```
3. Verify services are running:
   ```bash
   docker-compose ps
   ```

## Verification
### Check DCGM-Exporter Metrics
```bash
curl http://localhost:9400/metrics
```
You should see metrics prefixed with `nvidiagpuexporter_`.

### Check Prometheus Targets
Open Prometheus at http://localhost:9090/targets. Both `meeting-assistant` and `dcgm` jobs should be UP.

### Import Grafana Dashboard
1. Log into Grafana at http://localhost:3000 (admin/changeme).
2. Navigate to Dashboards > Import.
3. Upload `docker/config/grafana/dashboards/dcgm-gpu-monitoring.json`.
4. Select Prometheus as data source.
5. The dashboard will display GPU metrics.

## Dashboard Panels
1. **GPU Utilization** – Percentage of GPU compute usage.
2. **GPU Memory Used** – Amount of GPU memory used.
3. **GPU Temperature** – GPU temperature in Celsius.
4. **GPU Power Usage** – Power draw in watts.
5. **GPU SM Clock** – Streaming Multiprocessor clock frequency.
6. **GPU PCIe Throughput** – PCIe transmit/receive bandwidth.

## Troubleshooting
- If DCGM-Exporter fails to start, ensure NVIDIA Container Toolkit is installed and Docker has GPU access.
- Verify GPU passthrough with `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`.
- Check logs: `docker logs meeting-assistant-dcgm-exporter`.

## Next Steps
- TensorRT Optimization: Convert faster-whisper to TensorRT engine.
- Triton Deployment: Deploy Triton Inference Server with vLLM backend.

