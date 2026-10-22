#!/bin/bash
# A round's follower (round.py ROUND.json --follow --execute --watch MINUTES) run unattended on the workspace app. It
# runs under a supervisor in its own session (setsid nohup), so closing the terminal that started it does not stop it,
# and the supervisor starts it again where it exits while its round is recorded (a crash, the cloud unreachable while it
# planned); one runs per round (round.py's lock on this machine and its lease in the bucket); "install-boot" starts the
# recorded followers again when the app's Jupyter server starts; "stop" stops one and waits until it has exited. Never
# launched by an agent on its own: starting a follower is starting the round's submissions, the user's decision.
#
#   follow.sh start ROUND.json [MINUTES]   start the round's follower, every MINUTES (10 by default), unless one runs
#   follow.sh stop ROUND                   stop it (TERM: it ends its pass, releases the round, exits) and wait for it
#   follow.sh status                       each recorded round: its follower's pid or "not running", its log's last lines
#   follow.sh ensure                       start each recorded round's follower that is not running
#   follow.sh install-boot                 run "ensure" whenever the app's Jupyter server starts
#
# "start" keeps the workspace's settings from its environment (round.py's WORKSPACE_ENVIRONMENT, PROJECT_ID, REGION,
# STUDY_BUDGET) in STATE/<round>.env and records the round in STATE/rounds/<round>, which "ensure" and the supervisor
# read. A follower that ends (its round done, a STOP object) takes its round out of STATE/rounds, and so does "stop", so
# nothing starts it again; one that exits because another follower holds the round (round.py's status 4) is not started
# again by its supervisor. STATE is SVPGS_FOLLOW_STATE, ~/.svpgs-follow by default: the follower's lock, pid file and
# log (STATE/<round>.log, also copied to svpgs/_rounds/<round>/follow/follow.log), the supervisor's pid file
# (STATE/<round>.supervisor), and their output (STATE/<round>.out).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
STATE="${SVPGS_FOLLOW_STATE:-$HOME/.svpgs-follow}"
RESTART_SECONDS="${SVPGS_FOLLOW_RESTART_SECONDS:-60}"
CONTENDED=4
mkdir -p "$STATE/rounds"

alive() {
  # Process $1 runs and its command line holds $2 (a pid reused by another process is not it).
  kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qF -- "$2"
}

running() {
  local pid
  pid="$(cat "$STATE/$1.supervisor" 2>/dev/null)" && alive "$pid" "follow.sh supervise $1" && return 0
  pid="$(cat "$STATE/$1.pid" 2>/dev/null)" && alive "$pid" "$(cat "$STATE/$1.spec" 2>/dev/null || true)--follow" && return 0
  return 1
}

shown() {
  local pid
  pid="$(cat "$STATE/$1.pid" 2>/dev/null)" && alive "$pid" "--follow" && echo "$pid" || cat "$STATE/$1.supervisor"
}

supervise() {
  local round="$1" spec minutes status began wait="$RESTART_SECONDS" child=""
  local stopping=""
  echo "$$" > "$STATE/$round.supervisor"
  trap 'stopping=1; [ -z "$child" ] || kill -TERM "$child" 2>/dev/null || true' TERM INT
  while [ -f "$STATE/rounds/$round" ] && [ -z "$stopping" ]; do
    { read -r spec; read -r minutes; } < "$STATE/rounds/$round"
    began=$SECONDS
    python3 "$here/round.py" "$spec" --follow --execute --watch "$minutes" --state "$STATE" < /dev/null &
    child=$!
    # The pid at once: "stop" finds the follower while it builds its plan, before round.py writes it.
    echo "$child" > "$STATE/$round.pid"
    status=0
    wait "$child" || status=$?
    while kill -0 "$child" 2>/dev/null; do status=0; wait "$child" || status=$?; done
    child=""
    [ -f "$STATE/rounds/$round" ] && [ -z "$stopping" ] || break
    if [ "$status" -eq "$CONTENDED" ]; then
      echo "$(date '+%F %T %Z') round $round: another follower holds it; this one is not started again"
      break
    fi
    # Exited while its round is recorded: a crash or a failed plan. Started again, at once after a long run, else after
    # a wait that doubles up to 30 minutes.
    if [ $((SECONDS - began)) -gt 3600 ]; then wait="$RESTART_SECONDS"; fi
    echo "$(date '+%F %T %Z') round $round: the follower exited with status $status; started again in $wait s"
    sleep "$wait" & child=$!
    wait "$child" || true
    child=""
    wait=$((wait * 2 > 1800 ? 1800 : wait * 2))
  done
  rm -f "$STATE/$round.supervisor"
}

start() {
  local spec minutes round name
  spec="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  minutes="${2:-10}"
  [[ "$minutes" =~ ^[1-9][0-9]*$ ]] || { echo "MINUTES is a whole number" >&2; return 2; }
  round="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["round"])' "$spec")"
  if running "$round"; then
    echo "round $round: its follower already runs (pid $(shown "$round"))"
    return 0
  fi
  : > "$STATE/$round.env"
  for name in $(cd "$here" && python3 -c 'import round; print(" ".join(round.WORKSPACE_ENVIRONMENT))') PROJECT_ID REGION STUDY_BUDGET; do
    [ -z "${!name:-}" ] || printf 'export %s=%q\n' "$name" "${!name}" >> "$STATE/$round.env"
  done
  printf '%s\n%s\n' "$spec" "$minutes" > "$STATE/rounds/$round"
  printf '%s ' "$spec" > "$STATE/$round.spec"
  rm -f "$STATE/$round.pid" "$STATE/$round.supervisor"
  cd "$STATE"
  setsid nohup bash "$here/follow.sh" supervise "$round" >> "$STATE/$round.out" 2>&1 < /dev/null &
  local launched=$!
  for _ in $(seq 1 60); do
    if running "$round"; then
      echo "round $round: follower started (pid $(shown "$round")), every $minutes minutes; log $STATE/$round.log"
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
  local round="$1" pid supervisor
  rm -f "$STATE/rounds/$round"
  if ! running "$round"; then
    echo "round $round: no follower runs here"
    return 0
  fi
  supervisor="$(cat "$STATE/$round.supervisor" 2>/dev/null || true)"
  pid="$(cat "$STATE/$round.pid" 2>/dev/null || true)"
  [ -z "$supervisor" ] || ! alive "$supervisor" "follow.sh supervise $round" || kill -TERM "$supervisor" 2>/dev/null || true
  [ -z "$pid" ] || ! alive "$pid" "--follow" || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 300); do
    if ! running "$round"; then
      echo "round $round: follower ${pid:-$supervisor} stopped"
      return 0
    fi
    sleep 1
  done
  echo "round $round: follower ${pid:-$supervisor} did not stop in 5 minutes; it is still running" >&2
  return 1
}

case "${1:-}" in
  start) [ $# -ge 2 ] || { echo "usage: follow.sh start ROUND.json [MINUTES]" >&2; exit 2; }; start "$2" "${3:-10}" ;;
  stop) [ $# -eq 2 ] || { echo "usage: follow.sh stop ROUND" >&2; exit 2; }; stop "$2" ;;
  supervise) [ $# -eq 2 ] || exit 2; supervise "$2" ;;
  status)
    for file in "$STATE"/rounds/* "$STATE"/*.pid; do
      [ -e "$file" ] || continue
      round="$(basename "$file" .pid)"
      if running "$round"; then echo "round $round: running (pid $(shown "$round"))"; else echo "round $round: not running"; fi
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
