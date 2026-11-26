#!/usr/bin/env bash
# Start the SF0.1 (~0.1 GB) test run. Same options as run_sf.sh, e.g.
#   tools/scaleout/run_sf0_1.sh --detach
exec "$(dirname "$0")/run_sf.sh" sf0_1 "$@"
