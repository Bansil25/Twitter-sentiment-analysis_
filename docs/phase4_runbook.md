# Phase 4 Runbook — Docker, Kubernetes, Prometheus/Grafana

This is the "actually do it" guide. Every command is copy-paste ready.

---

## Part 1: Docker (you know basics — here's the layer above)

### Build the production image

```bash
# From project root
docker build -t sentimentai:latest .

# Check the size — should be around 1.5-2GB (mostly PyTorch + transformers)
docker images sentimentai:latest

# Compare builder vs runtime stages (proves the multi-stage win):
docker build --target builder -t sentimentai:builder .
docker images | grep sentimentai
# You'll see builder is significantly larger than runtime
```

### Run it locally (just the API, no extras)

```bash
docker run -d \
  -p 8000:8000 \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -e ENVIRONMENT=development \
  --name sentimentai \
  sentimentai:latest

# Tail the logs to watch it start
docker logs -f sentimentai

# Once you see "Application startup complete", test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict/public \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple just dropped the best product ever!"}'

# Cleanup
docker stop sentimentai && docker rm sentimentai
```

### Run the full stack with docker-compose

This is what you'll actually use day-to-day:

```bash
# Start everything (will take 2-3 min on first run)
docker compose up --build

# In another terminal, verify each service:
curl http://localhost:8000/health          # API
curl http://localhost:8000/metrics         # Prometheus metrics endpoint
open http://localhost:5000                  # MLflow UI (or xdg-open on Linux)
open http://localhost:9090                  # Prometheus UI
open http://localhost:3001                  # Grafana UI (admin/admin)

# Useful debug commands
docker compose ps                          # which services are up
docker compose logs api                    # logs for one service
docker compose logs -f api                 # follow logs
docker compose exec api bash               # shell inside the container
docker compose exec redis redis-cli        # Redis CLI

# Stop everything
docker compose down

# Stop AND wipe volumes (start fresh)
docker compose down -v
```

### The 4 Docker tricks that matter for interviews

1. **Layer ordering** — `COPY requirements.txt` BEFORE `COPY app/` so dependency
   installs are cached when only code changes.

2. **Multi-stage builds** — builder stage has gcc and 500MB of pip cache;
   runtime stage just has the installed packages. ~40% smaller final image.

3. **`.dockerignore`** — without this, you ship your `.git` history and trained
   model weights with the image. Bloats it from 1.5GB to 5GB+.

4. **Non-root user** — `USER appuser` in the Dockerfile. If anyone exploits a
   vuln in your app, they get appuser privileges instead of root.

---

## Part 2: Kubernetes — the actual commands

### Set up a local cluster (you don't need cloud yet)

Pick one of these (Mac/Linux):

```bash
# Option A: minikube — most popular, runs in a VM
brew install minikube
minikube start --cpus=4 --memory=8g
minikube addons enable ingress
minikube addons enable metrics-server     # required for HPA to work

# Option B: kind — lighter, runs in Docker
brew install kind
kind create cluster --name sentimentai

# Verify it's working
kubectl get nodes
kubectl get pods -A                       # all pods in all namespaces
```

### Deploy SentimentAI to K8s

```bash
# Apply manifests in order (the 00-, 01-, 02- numbering ensures order)
kubectl apply -f infrastructure/k8s/

# Watch the pods come up
kubectl get pods -n sentimentai -w        # -w = watch (Ctrl+C to exit)

# Once all are "Running" and "1/1 READY":
kubectl get all -n sentimentai

# Check logs of one pod
kubectl logs -n sentimentai deployment/sentimentai-api -f

# Open a port-forward to test
kubectl port-forward -n sentimentai svc/sentimentai-api 8000:8000
# In another terminal:
curl http://localhost:8000/health
```

### The 10 kubectl commands you'll actually use

```bash
# 1. Get things
kubectl get pods -n sentimentai                    # list pods
kubectl get deployments -n sentimentai             # list deployments
kubectl get svc -n sentimentai                     # list services

# 2. Describe (the most useful debug command)
kubectl describe pod <pod-name> -n sentimentai     # why is my pod stuck?

# 3. Logs
kubectl logs <pod-name> -n sentimentai             # one shot
kubectl logs -f deployment/sentimentai-api -n sentimentai  # follow

# 4. Exec into a pod (for debugging)
kubectl exec -it <pod-name> -n sentimentai -- bash

# 5. Port-forward (for accessing services without an Ingress)
kubectl port-forward -n sentimentai svc/sentimentai-api 8000:8000

# 6. Apply changes
kubectl apply -f infrastructure/k8s/03-api.yaml

# 7. Delete a pod (it'll auto-respawn)
kubectl delete pod <pod-name> -n sentimentai

# 8. Scale manually (HPA usually handles this)
kubectl scale deployment sentimentai-api --replicas=5 -n sentimentai

# 9. Rolling restart (e.g. after secret change)
kubectl rollout restart deployment sentimentai-api -n sentimentai

# 10. Check HPA status
kubectl get hpa -n sentimentai
```

### When things go wrong (they will)

| Symptom | Diagnostic |
|---|---|
| Pod stuck `Pending` | `kubectl describe pod <name>` — usually resource constraints |
| Pod `CrashLoopBackOff` | `kubectl logs <pod-name>` — read the actual error |
| `ImagePullBackOff` | Image name wrong, or registry needs auth |
| Service not routing | Check labels match: `kubectl get pods --show-labels` |
| HPA shows `<unknown>/70%` | metrics-server not installed |
| Pod `OOMKilled` | Memory limits too low — increase in `03-api.yaml` |

---

## Part 3: Prometheus & Grafana — the painless tour

### Mental model (this is all there is)

```
Your app                 Prometheus              Grafana
─────────                ──────────              ────────
Exposes                  Scrapes /metrics        Queries Prometheus
/metrics endpoint  ────► every 15 seconds  ────► via PromQL, draws charts
(already done!)          stores time-series
```

### Bring up the stack

```bash
docker compose up -d prometheus grafana
# (If you've already done docker compose up, they're already running)
```

### Step 1: Verify Prometheus sees your API

1. Open http://localhost:9090
2. Click **Status → Targets** in the top menu
3. You should see `sentimentai-api` with state **UP**
4. If it says **DOWN**: the API hasn't started yet, or there's a network issue.
   Run `docker compose logs prometheus` to check.

### Step 2: Try a PromQL query

In the Prometheus UI, paste any of these into the query bar and click Execute:

```promql
# Total request count
sum(sentimentai_requests_total)

# Requests per second over the last minute
sum(rate(sentimentai_requests_total[1m]))

# P95 latency in milliseconds
histogram_quantile(0.95, rate(sentimentai_request_latency_seconds_bucket[5m])) * 1000

# Error rate (5xx / total)
sum(rate(sentimentai_requests_total{status_code=~"5.."}[5m]))
/
sum(rate(sentimentai_requests_total[5m]))

# Predictions by sentiment
sum by (sentiment) (sentimentai_predictions_total)
```

Initially most queries return "no data" because you haven't generated any
traffic yet. Generate some:

```bash
# Generate load (run 200 requests)
for i in {1..200}; do
  curl -s -X POST http://localhost:8000/api/v1/predict/public \
    -H "Content-Type: application/json" \
    -d '{"text": "test message '$i'"}' > /dev/null
done
echo "Done"
```

Re-run the PromQL queries — now you have data.

### Step 3: View the pre-built Grafana dashboard

1. Open http://localhost:3001
2. Login: `admin` / `admin` (will prompt you to change)
3. Click **Dashboards** in the left sidebar
4. Click **SentimentAI — Production Overview**

You should see 8 panels showing real-time metrics:
- Requests per second
- P95 latency
- Error rate
- Median prediction confidence
- Latency percentiles over time
- Predictions by sentiment (stacked)
- Requests per endpoint
- Latency heatmap

Generate the load again and watch the charts update live.

### Step 4: Build your own panel (this is the skill)

In Grafana, click any panel → **Edit**. The "Query" tab is where you write
PromQL. Try this from scratch:

1. Dashboard → **New** → **Add visualization**
2. Choose **Prometheus** as the data source
3. In the query box, type: `sum(rate(sentimentai_predictions_total[1m]))`
4. Title it "Predictions per second"
5. Save

That's the entire Grafana workflow. Every dashboard is just panels, every
panel is just a PromQL query plus a visualization choice.

### The 4 metric types — what each one is for

| Type | Example | When to use |
|---|---|---|
| **Counter** | `sentimentai_requests_total` | Things that only go up (request count) |
| **Gauge** | `redis_connected_clients` | Things that go up AND down (queue depth) |
| **Histogram** | `sentimentai_request_latency_seconds` | Distributions (latency, sizes) |
| **Summary** | rarely used now | Like histogram but client-side computed |

Histograms are the most powerful — `histogram_quantile()` gives you P50/P95/P99
across all pods automatically. This is the right metric type for latency
EVERY time.

---

## Part 4: CI/CD — what changed from Phase 2

Your existing `.github/workflows/ci.yml` from Phase 2 is solid. The Phase 4
addition is **deploying to K8s** when tests pass on main. Here's the extra job
to add at the bottom of the existing file (deploy-k8s).

That's covered in the `.github/workflows/ci.yml` file in the zip.

---

## Recommended learning path

You said you need Kubernetes and Grafana/Prometheus. Here's the order:

1. **Week 1: get docker-compose running locally.**
   Generate traffic, watch Grafana update. Learn PromQL by clicking around.
   Read the comments in `prometheus.yml` and `03-api.yaml` until they feel obvious.

2. **Week 2: install minikube, deploy to it.**
   Run every kubectl command from the table above at least once.
   Intentionally break a pod (delete it, edit it badly) and watch K8s recover.

3. **Week 3: cloud deployment (Phase 5).**
   Use the same manifests against a real cluster — GKE Autopilot is the
   gentlest start ($0.10/hour for a small cluster).

4. **Interview-ready by week 4.**
   You'll be able to answer "walk me through your deployment" with real
   experience, not memorized concepts.
