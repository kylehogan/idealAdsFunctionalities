#!/bin/bash

# Bash script to run distinguishing_game.py for specified plot types
# Usage: ./reproduce_plots.sh [--small] [--plots-only] [--cores N] [plot_type1 plot_type2 ...]
# Options:
#   --small         Use small number of trials for quick runs
#   --plots-only    Only generate plots using existing data (do not rerun experiments)
#   --cores N       Number of CPU cores to use (default: 8)
#   plot_type       One or more plot types to run: targeting, engagement, epsilon, private_v_nonprivate (default: all)
#   --help          Show this help message and exit

SCRIPT="distinguishing_game.py"

SMALL_FLAG=""
PLOTS_ONLY_FLAG=""
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
        --small)
            SMALL_FLAG="--trials 10"
            shift
            ;;
        --plots-only)
            PLOTS_ONLY_FLAG="--plots_only"
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
    # Always clean unless --plots-only is set
    CLEAN_FLAG=""
    if [ -z "$PLOTS_ONLY_FLAG" ]; then
        CLEAN_FLAG="--clean"
    fi

    # Set trials to 100 for private_v_nonprivate or epsilon
    TRIALS_FLAG=""
    if [[ "$plot_type" == "private_v_nonprivate" || "$plot_type" == "epsilon" ]]; then
        TRIALS_FLAG="--trials 100"
    else
        TRIALS_FLAG="$SMALL_FLAG"
    fi

    echo "Running: $SCRIPT --plot_type $plot_type $CLEAN_FLAG $TRIALS_FLAG $PLOTS_ONLY_FLAG --cores $CORES"
    python $SCRIPT $CLEAN_FLAG $TRIALS_FLAG $PLOTS_ONLY_FLAG --cores $CORES --plot_type $plot_type 
done