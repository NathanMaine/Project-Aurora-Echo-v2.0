# LAN Deployment & Access Guide

Use this guide to expose the Real-Time AI Meeting Assistant to trusted devices on your
home network via Docker and Traefik.

## Prerequisites
- Ubuntu 22.04 LTS (or newer) with the latest NVIDIA driver and CUDA runtime.
- `docker` and `docker compose` installed.
- `nvidia-container-toolkit` configured (`sudo apt install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure && sudo systemctl restart docker`).
- DNS or hosts entry pointing `assistant.lan` (example) to the workstation IP, or share the raw IP with the family.

## One-Time Setup
1. Duplicate `.env.docker` to store secrets securely (e.g., load from a password manager).
2. Generate TLS materials (self-signed example):
   ```bash
   mkdir -p docker/certs
   openssl req -x509 -nodes -days 825 -newkey rsa:4096 \
     -keyout docker/certs/lan.key \
     -out docker/certs/lan.crt \
     -subj "/CN=assistant.lan"
   chmod 600 docker/certs/lan.key
   ```
   Install `lan.crt` on client devices so browsers trust the certificate.
3. (Optional) Update `docker/config/traefik.yml` with new basic-auth credentials:
   ```bash
   htpasswd -nbB family newpassword
   ```
   Replace the default hash in the file.

## Launching the Stack
```bash
docker compose --profile model up --build -d
```
- `--profile model` starts the bundled `vllm` container. Omit it if you want to
  target an external model endpoint.
- Add `PROMETHEUS_MULTIPROC_DIR=/tmp/prom` to `.env.docker` if you want multi-process metrics.

## Stopping / Updating
```bash
docker compose down        # stop
docker compose pull         # update images
```
Rebuild the API container after code changes:
```bash
docker compose build api && docker compose up -d api
```

## Access for Family Members
- URL: `https://assistant.lan/` (or `https://<workstation-ip>/`).
- Credentials: default `family / welcome123` (change in Traefik config!).
- Browsers will warn about self-signed certificates until the LAN CA is installed.

## SSH Tunnel Alternative
For off-LAN laptops, tunnel via:
```bash
ssh -L 8443:localhost:443 user@workstation
```
Then browse `https://localhost:8443/`.

## Monitoring
- Prometheus: `http://<workstation-ip>:9090`
- Grafana: `http://<workstation-ip>:3000` (default `admin/changeme`)
- API metrics endpoint: `https://assistant.lan/metrics`

Dashboards to build:
- Inference latency (`meeting_assistant_inference_job_duration_seconds`)
- Queue depth (`meeting_assistant_inference_queue_depth{backend="docker"}`)
- ASR / LLM / diarization latency histograms

## Security Checklist
- Rotate all API keys regularly and store them outside Git (secret manager).
- Replace default passwords (Traefik basic auth, Grafana admin).
- Limit firewall exposure to trusted LAN/VPN segments.
- Review Traefik logs for unexpected requests.
- Consider enabling ACME/Let’s Encrypt if you eventually expose the service beyond the LAN.

## Future Enhancements
- Swap Traefik self-signed certs for ACME automation (requires DNS challenge or public domain).
- Externalise the inference queue via Redis/FastStream containers for multi-node GPU scaling.
- Integrate SSO (Authelia/Keycloak) if you deploy to a wider audience.
