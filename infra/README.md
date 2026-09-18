# infra

| File | Purpose |
|---|---|
| `docker-compose.yml` | Dev dependencies only: Postgres 16, Redis 7, MinIO (S3), Ollama. App services run from source. |
| `compose.prod.yml` | (Phase 5) single-VPS production: all services + Caddy TLS. |
| `scripts/` | `backup.sh`, `restore.sh`, `rotate_keys.md` (Phase 5). |

Rules: no application code here; no secrets here (compose reads `.env`); every service has a healthcheck.
