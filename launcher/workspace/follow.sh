#!/bin/bash
# A round's follower (round.py ROUND.json --follow --execute --watch MINUTES) run unattended on the workspace app. It
# runs in its own session (setsid nohup), so closing the terminal that started it does not stop it; one runs per round
# (round.py's lock on this machine and its lease in the bucket); "install-boot" starts the recorded followers again when
# the app's Jupyter server starts; "stop" stops one and waits until it has exited. Never launched by an agent on its
# own: starting a follower is starting the round's submissions, the user's decision.
#
#   follow.sh start ROUND.json [MINUTES]   start the round's follower, every MINUTES (10 by default), unless one runs
#   follow.sh stop ROUND                   stop it (TERM: it ends its pass, releases the round, exits) and wait for it
#   follow.sh status                       each recorded round: its follower's pid or "not running", its log's last lines
#   follow.sh ensure                       start each recorded round's follower that is not running
#   follow.sh install-boot                 run "ensure" whenever the app's Jupyter server starts
#
# "start" keeps the workspace's settings from its environment (round.py's WORKSPACE_ENVIRONMENT, PROJECT_ID, REGION,
# STUDY_BUDGET) in STATE/<round>.env and records the round in STATE/rounds/<round>, which "ensure" reads. A follower
# that ends (its round done, a STOP object, a stop) takes its round out of STATE/rounds, so nothing starts it again.
# STATE is SVPGS_FOLLOW_STATE, ~/.svpgs-follow by default: the follower's lock, pid file and log (STATE/<round>.log,
# also copied to svpgs/_rounds/<round>/follow/follow.log), and its output (STATE/<round>.out).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
STATE="${SVPGS_FOLLOW_STATE:-$HOME/.svpgs-follow}"
mkdir -p "$STATE/rounds"

running() {
  local pid
  pid="$(cat "$STATE/$1.pid" 2>/dev/null)" || return 1
  kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "round.py .*--follow"
}

start() {
  local spec minutes round name
  spec="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  minutes="${2:-10}"
  [[ "$minutes" =~ ^[1-9][0-9]*$ ]] || { echo "MINUTES is a whole number" >&2; return 2; }
  round="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["round"])' "$spec")"
  if running "$round"; then
    echo "round $round: its follower already runs (pid $(cat "$STATE/$round.pid"))"
    return 0
  fi
  : > "$STATE/$round.env"
  for name in $(cd "$here" && python3 -c 'import round; print(" ".join(round.WORKSPACE_ENVIRONMENT))') PROJECT_ID REGION STUDY_BUDGET; do
    [ -z "${!name:-}" ] || printf 'export %s=%q\n' "$name" "${!name}" >> "$STATE/$round.env"
  done
  printf '%s\n%s\n' "$spec" "$minutes" > "$STATE/rounds/$round"
  rm -f "$STATE/$round.pid"
  # The plan is built (minutes of reads of the bucket) before the follower takes the round and writes its pid file.
  cd "$STATE"
  setsid nohup python3 "$here/round.py" "$spec" --follow --execute --watch "$minutes" --state "$STATE" \
    >> "$STATE/$round.out" 2>&1 < /dev/null &
  local launched=$!
  for _ in $(seq 1 600); do
    if running "$round"; then
      echo "round $round: follower started (pid $(cat "$STATE/$round.pid")), every $minutes minutes; log $STATE/$round.log"
      return 0
    fi
    kill -0 "$launched" 2>/dev/null || break
    sleep 1
  done
  echo "round $round: the follower did not start; its output:" >&2
  tail -n 20 "$STATE/$round.out" >&2
  return 1
}

stop() {
  local round="$1" pid
  rm -f "$STATE/rounds/$round"
  if ! running "$round"; then
    echo "round $round: no follower runs here"
    return 0
  fi
  pid="$(cat "$STATE/$round.pid")"
  kill -TERM "$pid"
  for _ in $(seq 1 300); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "round $round: follower $pid stopped"
      return 0
    fi
    sleep 1
  done
  echo "round $round: follower $pid did not stop in 5 minutes; it is still running" >&2
  return 1
}

case "${1:-}" in
  start) [ $# -ge 2 ] || { echo "usage: follow.sh start ROUND.json [MINUTES]" >&2; exit 2; }; start "$2" "${3:-10}" ;;
  stop) [ $# -eq 2 ] || { echo "usage: follow.sh stop ROUND" >&2; exit 2; }; stop "$2" ;;
  status)
    for file in "$STATE"/rounds/* "$STATE"/*.pid; do
      [ -e "$file" ] || continue
      round="$(basename "$file" .pid)"
      if running "$round"; then echo "round $round: running (pid $(cat "$STATE/$round.pid"))"; else echo "round $round: not running"; fi
      tail -n 3 "$STATE/$round.log" 2>/dev/null | sed 's/^/  /'
    done | awk '!seen[$0]++' ;;
  ensure)
    for file in "$STATE"/rounds/*; do
      [ -f "$file" ] || continue
      round="$(basename "$file")"
      running "$round" && continue
      { read -r spec; read -r minutes; } < "$file"
      (set -a && . "$STATE/$round.env" && set +a && start "$spec" "$minutes") || echo "round $round: not started" >&2
    done ;;
  install-boot)
    config="$HOME/.jupyter/jupyter_server_config.py"
    if grep -q "svpgs-follow" "$config" 2>/dev/null; then
      echo "already installed in $config"
    else
      mkdir -p "$(dirname "$config")"
      cat >> "$config" <<PY

# svpgs-follow: start the study's recorded round followers when this server starts (launcher/workspace/follow.sh ensure).
try:
    import subprocess as _svpgs_follow
    _svpgs_follow.Popen(["bash", "$here/follow.sh", "ensure"], start_new_session=True, stdin=_svpgs_follow.DEVNULL,
                        stdout=_svpgs_follow.DEVNULL, stderr=_svpgs_follow.DEVNULL)
except Exception:
    pass
PY
      echo "installed in $config"
    fi ;;
  *) echo "usage: follow.sh start ROUND.json [MINUTES] | stop ROUND | status | ensure | install-boot" >&2; exit 2 ;;
esac
