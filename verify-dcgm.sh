#!/bin/bash
# Verification script for DCGM integration

set -e

echo "=== Checking DCGM-Exporter service ==="
if ! docker ps | grep -q meeting-assistant-dcgm-exporter; then
    echo "DCGM-Exporter container not running. Starting stack..."
    docker-compose up -d dcgm-exporter
    sleep 5
fi

echo "=== Testing metrics endpoint ==="
curl -s http://localhost:9400/metrics | head -20

echo "=== Checking Prometheus targets ==="
curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | .labels.job + ": " + .health' | grep dcgm

echo "=== Grafana dashboard ==="
echo "Dashboard JSON: docker/config/grafana/dashboards/dcgm-gpu-monitoring.json"
echo "Import manually via Grafana UI."

