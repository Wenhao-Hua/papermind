#!/usr/bin/env bash
# Update the live PaperMind service on the deploy host (the box that runs
# `papermind serve` behind the cloudflared tunnel) to the latest main.
#
# On the host:                bash /opt/papermind/scripts/deploy.sh
# From your machine:          ssh root@HOST 'bash -s' < scripts/deploy.sh
#
# Override locations via env: PAPERMIND_REPO, PAPERMIND_SERVICE.
set -euo pipefail

REPO="${PAPERMIND_REPO:-/opt/papermind}"
SERVICE="${PAPERMIND_SERVICE:-papermind}"

echo "[deploy] fetching latest main into $REPO"
git -C "$REPO" fetch --depth 1 origin main
git -C "$REPO" reset --hard FETCH_HEAD
echo "[deploy] now at $(git -C "$REPO" rev-parse --short HEAD)  $(git -C "$REPO" log -1 --format=%s)"

echo "[deploy] syncing dependencies (editable install)"
"$REPO/.venv/bin/pip" install -q -e "$REPO[web]"

echo "[deploy] restarting $SERVICE"
systemctl restart "$SERVICE"
sleep 2
if systemctl is-active --quiet "$SERVICE"; then
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ || echo "000")
  echo "[deploy] $SERVICE active · local homepage HTTP $code"
else
  echo "[deploy] FAILED to start $SERVICE:" >&2
  systemctl status "$SERVICE" --no-pager -l | head -20 >&2
  exit 1
fi
