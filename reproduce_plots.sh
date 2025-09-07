#!/bin/bash

# Bash script to run distinguishing_game.py for specified plot types
# Usage: ./reproduce_plots.sh [--clean] [--small] [--cores N] [plot_type1 plot_type2 ...]
# Options:
#   --clean         Remove previous data for each plot type before running
#   --small         Use small number of trials for quick runs
#   --cores N       Number of CPU cores to use (default: 8)
#   plot_type       One or more plot types to run: targeting, engagement, epsilon, private_v_nonprivate (default: all)
#   --help          Show this help message and exit

SCRIPT="distinguishing_game.py"

CLEAN_FLAG=""
SMALL_FLAG=""
CORES=8
PLOT_TYPES=()

# Help message
if [[ "$1" == "--help" ]]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN_FLAG="--clean"
            shift
            ;;
        --small)
            SMALL_FLAG="--trials 100"
            shift
            ;;
        --cores)
            CORES="$2"
            shift 2
            ;;
        --cores=*)
            CORES="${1#*=}"
            shift
            ;;
        *)
            PLOT_TYPES+=("$1")
            shift
            ;;
    esac
done

# Default plot types if none specified
if [ ${#PLOT_TYPES[@]} -eq 0 ]; then
    PLOT_TYPES=("private_v_nonprivate" "targeting" "epsilon" "engagement")
fi

for plot_type in "${PLOT_TYPES[@]}"
do
    echo "Running plot_type: $plot_type $CLEAN_FLAG $SMALL_FLAG --cores $CORES"
    python $SCRIPT --plot_type $plot_type $CLEAN_FLAG $SMALL_FLAG --cores $CORES
done