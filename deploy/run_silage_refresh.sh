#!/usr/bin/env bash
# run_silage_refresh.sh: refresh the NY corn-silage data the Silage tab ships with. Run by cron on the JSA droplet through
# /opt/alerting/cron-alert: daily Jan-Mar (NASS publishes the state silage estimates in mid-January and the county estimates
# in Feb-Mar) and Tuesdays the rest of the year. The droplet is /opt/soy-crush-calculator, a clone of this repo.
#
# It pulls, asks NASS QuickStats for NY corn-silage rows (build_ny_silage.py --if-changed) and, only when NASS's rows differ from
# ny_silage_snapshot.json, commits that one file and pushes it to main. Streamlit Cloud then redeploys; the app reads the JSON
# on every run, so no Reboot is needed for a data-only commit. A reply that looks partial makes the build exit 1, which fires
# cron-alert's failure email and leaves the committed file alone.
#
# NASS_API_KEY is read from /opt/usda-nass-etl/.env (one copy to rotate) into this process only, never into the repo or the log.
# Everything is inside main() so a pull that rewrites this script mid-run cannot make bash execute half-old, half-new lines.
# SILAGE_NO_PUSH=1 commits locally and skips the push (for testing).
set -euo pipefail

main() {
  local REPO=/opt/soy-crush-calculator
  local PY="$REPO/.venv/bin/python"
  local ENV_FILE=/opt/usda-nass-etl/.env
  mkdir -p "$REPO/logs"
  exec >> "$REPO/logs/silage_refresh_$(date +%F).log" 2>&1
  echo "[$(date '+%F %T')] NY silage refresh"
  exec 9> "$REPO/logs/.silage_refresh.lock"
  flock -n 9 || { echo "another refresh is still running"; return 0; }

  cd "$REPO"
  # A deploy clone: only this job commits here, so start from GitHub's main (an unpushed commit from a failed push is regenerated).
  git fetch -q origin main
  git reset -q --hard origin/main

  local key
  key=$(sed -n 's/^NASS_API_KEY=//p' "$ENV_FILE" | head -n 1 | tr -d "\"'\r")
  [ -n "$key" ] || { echo "no NASS_API_KEY in $ENV_FILE"; return 1; }
  NASS_API_KEY="$key" "$PY" build_ny_silage.py --if-changed

  if git diff --quiet -- ny_silage_snapshot.json; then
    echo "nothing to commit"
    return 0
  fi
  git config user.email '275148418+koltenpostin93-blip@users.noreply.github.com'
  git config user.name 'JSA Droplet'
  git add ny_silage_snapshot.json
  git commit -q -m "data: refresh NY corn-silage snapshot from NASS ($(date +%F))"
  if [ "${SILAGE_NO_PUSH:-0}" = 1 ]; then
    echo "SILAGE_NO_PUSH=1: committed locally, not pushed"
    return 0
  fi
  git push -q origin HEAD:main || { git pull -q --rebase origin main && git push -q origin HEAD:main; }
  echo "pushed: $(git log --oneline -1)"
}

main "$@"
exit 0
