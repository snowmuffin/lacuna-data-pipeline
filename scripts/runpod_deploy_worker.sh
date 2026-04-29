#!/bin/bash
# ============================================================
# lacuna-data-pipeline — RunPod GPU pod: REST API (8000) + Jupyter (8888)
#
# API (scripts/data_worker_api.py):
#   GET  /health
#   POST /classify           JSON { text }  → categories [..], category comma-joined (multi-label)
#   POST /classify/batch     JSON { items: [{ text }, ...] }  → identity|values|persona|general
#   POST /translate, /translate/batch  → 501 (classification-only worker)
# Optional header: X-API-Key when LACUNA_DATA_API_KEY is set on the pod.
#
# Defaults: CLASSIFY_MODEL=Qwen2.5-7B-Instruct, PRELOAD_CLASSIFY=1, GPU_TYPE=A100 80GB PCIe with an extended fallback chain (see GPU_FALLBACKS).
# Usage:
#   ./scripts/runpod_deploy_worker.sh
#
# Secrets: .env.example → .env.local (preferred) or .env at repo root
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_LOCAL="$REPO_ROOT/.env.local"
ENV_DOT="$REPO_ROOT/.env"

# Preserve POD_NAME if the caller already set it (e.g. notebook multi-pod: POD_NAME=stem-2).
# Sourcing .env below would otherwise overwrite with a single default name.
CALLER_POD_NAME="${POD_NAME:-}"

if [ -f "$ENV_LOCAL" ]; then
    echo "📄 Loading secrets from .env.local..."
    set -a
    # shellcheck source=/dev/null
    source "$ENV_LOCAL"
    set +a
elif [ -f "$ENV_DOT" ]; then
    echo "📄 Loading secrets from .env..."
    set -a
    # shellcheck source=/dev/null
    source "$ENV_DOT"
    set +a
else
    echo "⚠️  No .env.local or .env under repo root. Using exported environment variables only."
fi

RUNPOD_API_KEY="${RUNPOD_API_KEY:?❌ Set RUNPOD_API_KEY in .env.local (or .env) or export it}"
GITHUB_TOKEN="${GITHUB_TOKEN:?❌ Set GITHUB_TOKEN in .env.local (or .env) or export it}"

# Optional — forwarded to the pod for data_worker_api (REST on port 8000)
LACUNA_DATA_API_KEY="${LACUNA_DATA_API_KEY:-}"
# Stronger default for classification quality (override in .env if needed)
CLASSIFY_MODEL="${CLASSIFY_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PRELOAD_CLASSIFY="${PRELOAD_CLASSIFY:-1}"

# Pod settings — default GPU targets higher VRAM/throughput for 7B classify
POD_NAME="${POD_NAME:-lacuna-data-worker}"
if [ -n "$CALLER_POD_NAME" ]; then
    POD_NAME="$CALLER_POD_NAME"
fi
GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"
# Order: higher typical availability / lower cost first, then larger VRAM / premium (must match RunPod gpuTypeId strings).
GPU_FALLBACKS="${GPU_FALLBACKS:-NVIDIA A40,NVIDIA RTX 4090,NVIDIA RTX A5000,NVIDIA L4,NVIDIA RTX PRO 4500,NVIDIA RTX 3090,NVIDIA RTX 6000 Ada,NVIDIA L40S,NVIDIA RTX A6000,NVIDIA RTX PRO 6000,NVIDIA A100 80GB SXM}"
GPU_COUNT="${GPU_COUNT:-1}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
VOLUME_SIZE="${VOLUME_SIZE:-80}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-change-me}"

REPO_OWNER="${REPO_OWNER:?❌ Set REPO_OWNER to your GitHub org or username}"
REPO_NAME="${REPO_NAME:-lacuna-data-pipeline}"

HTTP_PORTS="8000/http,8888/http"

read -r -d '' STARTUP_SCRIPT << 'STARTUP_EOF' || true
#!/bin/bash
set -e
echo "🚀 [lacuna-data-pipeline] Worker pod startup..."

cd /workspace
REPO="__REPO_NAME__"
if [ ! -d "$REPO" ]; then
    echo "📦 Cloning $REPO ..."
    git clone https://x-access-token:${GITHUB_TOKEN}@github.com/__REPO_OWNER__/__REPO_NAME__.git
else
    echo "📂 Repository exists, pulling latest..."
    cd "$REPO" && git pull origin main || true && cd /workspace
fi

DATA_DIR="/workspace/$REPO"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Expected data root at $DATA_DIR"
    exit 1
fi

echo "📝 Writing .env from pod environment..."
umask 077
cat > "$DATA_DIR/.env" << ENVEOF
# Generated on pod boot by runpod_deploy_worker.sh
GITHUB_TOKEN=${GITHUB_TOKEN}
HF_TOKEN=${HF_TOKEN:-}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
LACUNA_DATA_API_KEY=${LACUNA_DATA_API_KEY:-}
CLASSIFY_MODEL=${CLASSIFY_MODEL:-}
PRELOAD_CLASSIFY=${PRELOAD_CLASSIFY:-0}
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
ENVEOF
echo "✅ $DATA_DIR/.env written (mode 600)"

echo "📦 Installing worker / HF stack..."
pip install --upgrade pip -q
pip install -r "$DATA_DIR/requirements-runpod.txt" 2>&1 | tail -15
echo "✅ requirements-runpod.txt installed"

echo "🌐 Starting data worker API (FastAPI) on port 8000..."
cd "$DATA_DIR/scripts"
# Sync CUDA errors for accurate stack traces on device asserts (slower; unset or 0 to disable).
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
nohup uvicorn data_worker_api:app --host 0.0.0.0 --port 8000 \
    > /workspace/data-worker-api.log 2>&1 &
echo "✅ API — docs: /docs — log: /workspace/data-worker-api.log"

echo "📓 Starting Jupyter Lab on port 8888..."
pip install -q jupyterlab
nohup jupyter lab \
    --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN}" \
    --ServerApp.allow_remote_access=True \
    --notebook-dir="$DATA_DIR" \
    > /workspace/jupyter.log 2>&1 &
echo "✅ Jupyter Lab — notebook-dir=$DATA_DIR (log: /workspace/jupyter.log)"

echo "🟢 Pod ready. Run batches from notebooks or CLI under $DATA_DIR"
sleep infinity
STARTUP_EOF

STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_OWNER__/$REPO_OWNER}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_NAME__/$REPO_NAME}"

ENCODED_STARTUP=$(echo "$STARTUP_SCRIPT" | base64 | tr -d '\n')

GPU_ARR=("$GPU_TYPE")
old_ifs="$IFS"
IFS=','
for gpu in $GPU_FALLBACKS; do
    gpu=$(printf '%s' "$gpu" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -n "$gpu" ] && [ "$gpu" != "$GPU_TYPE" ] && GPU_ARR+=("$gpu")
done
IFS="$old_ifs"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠️  HF_TOKEN is empty — set it in .env.local / .env for gated models and reliable Hub downloads."
fi

echo ""
echo "🚀 Creating RunPod pod (lacuna-data-pipeline worker)"
echo "   Name:      $POD_NAME"
echo "   GPU:       $GPU_TYPE x$GPU_COUNT (fallbacks: $GPU_FALLBACKS)"
echo "   Image:     $CONTAINER_IMAGE"
echo "   Storage:   ${VOLUME_SIZE}GB → /workspace"
echo "   REST API:  port 8000 (classify; translate disabled)"
echo "   Jupyter:   port 8888, token=$JUPYTER_TOKEN"
echo ""

POD_ID="UNKNOWN"
RESPONSE=""

for TRY_GPU in "${GPU_ARR[@]}"; do
    echo "🔄 Trying GPU: $TRY_GPU"
    # Build GraphQL with JSON-escaped env values so HF_TOKEN quotes/newlines cannot break the mutation.
    QUERY="$(
        POD_NAME="$POD_NAME" \
        CONTAINER_IMAGE="$CONTAINER_IMAGE" \
        TRY_GPU="$TRY_GPU" \
        GPU_COUNT="$GPU_COUNT" \
        CLOUD_TYPE="$CLOUD_TYPE" \
        VOLUME_SIZE="$VOLUME_SIZE" \
        HTTP_PORTS="$HTTP_PORTS" \
        ENCODED_STARTUP="$ENCODED_STARTUP" \
        GITHUB_TOKEN="$GITHUB_TOKEN" \
        HF_TOKEN="${HF_TOKEN:-}" \
        ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
        JUPYTER_TOKEN="$JUPYTER_TOKEN" \
        LACUNA_DATA_API_KEY="${LACUNA_DATA_API_KEY:-}" \
        LACUNA_DATA_VERBOSE_ERRORS="${LACUNA_DATA_VERBOSE_ERRORS:-}" \
        CLASSIFY_MODEL="${CLASSIFY_MODEL:-}" \
        PRELOAD_CLASSIFY="$PRELOAD_CLASSIFY" \
        python3 - <<'PY'
import json, os

def esc(s: str) -> str:
    return json.dumps(s)

vol = int(os.environ["VOLUME_SIZE"])
gpu_count = int(os.environ["GPU_COUNT"])
cloud = os.environ["CLOUD_TYPE"]
enc = os.environ["ENCODED_STARTUP"]
docker_args = (
    "bash -c 'echo " + enc + " | base64 -d > /workspace/startup.sh && bash /workspace/startup.sh'"
)
env_pairs = [
    ("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", "")),
    ("HF_TOKEN", os.environ.get("HF_TOKEN", "")),
    ("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")),
    ("JUPYTER_TOKEN", os.environ.get("JUPYTER_TOKEN", "")),
    ("LACUNA_DATA_API_KEY", os.environ.get("LACUNA_DATA_API_KEY", "")),
    ("LACUNA_DATA_VERBOSE_ERRORS", os.environ.get("LACUNA_DATA_VERBOSE_ERRORS", "")),
    ("CLASSIFY_MODEL", os.environ.get("CLASSIFY_MODEL", "")),
    ("PRELOAD_CLASSIFY", os.environ.get("PRELOAD_CLASSIFY", "0")),
]
env_block = ",\n        ".join(
    "{{ key: {}, value: {} }}".format(esc(k), esc(v)) for k, v in env_pairs
)
q = f"""mutation {{
  podFindAndDeployOnDemand(
    input: {{
      name: {esc(os.environ["POD_NAME"])}
      imageName: {esc(os.environ["CONTAINER_IMAGE"])}
      gpuTypeId: {esc(os.environ["TRY_GPU"])}
      gpuCount: {gpu_count}
      cloudType: {cloud}
      volumeInGb: {vol}
      volumeMountPath: {esc("/workspace")}
      containerDiskInGb: 100
      ports: {esc(os.environ["HTTP_PORTS"])}
      dockerArgs: {esc(docker_args)}
      env: [
        {env_block}
      ]
    }}
  ) {{
    id
    name
    desiredStatus
    imageName
    machine {{
      gpuDisplayName
    }}
  }}
}}"""
print(q, end="")
PY
    )"

    RESPONSE=$(curl -s -X POST \
      "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{\"query\": $(echo "$QUERY" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}")

    if echo "$RESPONSE" | grep -q "SUPPLY_CONSTRAINT"; then
        echo "   ⚠️  $TRY_GPU not available, trying next..."
        continue
    fi

    POD_ID=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pod = data.get('data', {}).get('podFindAndDeployOnDemand', {})
print(pod.get('id', 'UNKNOWN'))
" 2>/dev/null || echo "UNKNOWN")

    if [ "$POD_ID" != "UNKNOWN" ] && [ -n "$POD_ID" ]; then
        echo "   ✅ Got pod with $TRY_GPU"
        break
    fi

    echo ""
    echo "📡 RunPod API Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    echo ""
    echo "❌ Pod creation failed."
    exit 1
done

echo ""
echo "📡 RunPod API Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

if [ "$POD_ID" != "UNKNOWN" ] && [ -n "$POD_ID" ]; then
    echo ""
    echo "✅ Pod created"
    echo "   Pod ID:  $POD_ID"
    echo ""
    echo "🔗 Access"
    echo "   Console:  https://www.runpod.io/console/pods"
    echo "   REST API: https://${POD_ID}-8000.proxy.runpod.net/docs"
    echo "   Jupyter:  https://${POD_ID}-8888.proxy.runpod.net/lab?token=$JUPYTER_TOKEN"
    echo "   SSH:      ssh ${POD_ID}@ssh.runpod.io"
    echo ""
    echo "📂 On the pod: repository at /workspace/$REPO_NAME"
else
    echo ""
    echo "❌ No capacity for GPUs: ${GPU_ARR[*]}"
    exit 1
fi
