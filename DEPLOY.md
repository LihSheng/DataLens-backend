# Phoenix Observability — Deployment Guide

## Final Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser                                                    │
│     │                                                       │
│     ├── https://your-app.com/observability  (admin only)    │
│     │         │                                            │
│     │         └──► Frontend (React)                        │
│     │                    │                                  │
│     │                    ▼                                  │
│     │              Backend API (FastAPI)                    │
│     │                    │                                  │
│     │                    ├── /api/phoenix/*  (admin only)   │
│     │                    └──► OTel traces                   │
│     │                                                       │
│     └── https://phoenix.your-domain.com   (Cloudflare ZT)  │
│                 │                                            │
│                 └──► Phoenix :6006  (localhost only)        │
└─────────────────────────────────────────────────────────────┘

Services:   Frontend ←→ API ←→ Postgres / Redis / Phoenix
Network:    Docker bridge "datalens" (isolated)
```

---

## Service Ports

| Service | Host | Purpose |
|---|---|---|
| API | `0.0.0.0:8000` | FastAPI backend |
| Postgres | `127.0.0.1:5432` | DB (localhost only) |
| Redis | `127.0.0.1:6379` | Cache (localhost only) |
| Phoenix | `127.0.0.1:6006` | Traces UI (localhost only — no direct access) |

---

## Phoenix Access — Cloudflare Zero Trust

Phoenix is intentionally **not exposed** to the internet. All traffic routes through the backend proxy (`/api/phoenix/*`) which requires admin authentication.

### Option 1: Cloudflare Zero Trust Tunnel (recommended)

1. Create a Cloudflare Zero Trust account at [dash.cloudflare.com](https://dash.cloudflare.com)
2. Create a tunnel pointing to `http://phoenix:6006` (Docker internal DNS)
3. Give the tunnel a public hostname, e.g. `phoenix.your-domain.com`
4. Enable **Access Policy** → require Google Workspace / GitHub / Microsoft SSO
5. In `docker-compose.yml`, uncomment the `cloudflared` block and set `CLOUDFLARED_TOKEN` in your `.env`:
   ```env
   CLOUDFLARED_TOKEN=your-tunnel-token-here
   ```

**Result:** `https://phoenix.your-domain.com` → Phoenix with full identity-aware access control.

### Option 2: SSH Tunnel (dev only)

```bash
ssh -L 6006:localhost:6006 user@your-server
# Then open http://localhost:6006 in your browser
```

---

## Environment Variables

Ensure these are set in your `.env`:

```env
# Phoenix OTel collector
PHOENIX_BASE_URL=http://phoenix:6006
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces

# OTel export
OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006
```

---

## OTel Trace Flow

1. User sends a chat message → `POST /api/chat`
2. FastAPI creates a root span (`rag_request`)
3. RAG chain adds child spans (retrieval, rerank, generation)
4. `run_live_ragas_eval()` fires async → scores written to Phoenix as span annotations
5. User submits feedback → `POST /api/feedback` → `submit_feedback()` annotation posted to Phoenix
6. Frontend `/observability` tab polls `GET /api/phoenix/traces` → admin-only

---

## Quick Start

```bash
# 1. Copy and fill in your env (Docker compose)
cp .env.docker.example .env

# 2. Start everything
docker compose up -d

# 3. Verify Phoenix is running (local only)
curl http://localhost:6006

# 4. Set up Cloudflare tunnel (optional — see above)

# 5. Open the app
open http://localhost:3000
```

---

## Switching to Arize Cloud (optional, for production)

If you outgrow self-hosted Phoenix:

1. Create an account at [app.arize.com](https://app.arize.com)
2. Change your OTel exporter endpoint:
   ```env
   OTEL_EXPORTER_OTLP_ENDPOINT=https://app.arize.com/v1/traces
   PHOENIX_API_KEY=your-arize-api-key
   ```
3. The self-hosted Phoenix container is no longer needed
4. Update `PHOENIX_BASE_URL` to Arize Cloud in your env
