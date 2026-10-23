#!/bin/bash
# The wide screen's after-script (FASTPGS_SCREEN_PRESPEC.txt): aou_fastpgs.sh fitting LDpred2-auto alone on both arms.
set -euo pipefail
exec bash "$(dirname "$0")/aou_fastpgs.sh" --methods ldpred2_auto
