#!/bin/bash

# Bash script to run distinguishing_game.py for all plot types
# Usage: ./reproduce_plots.sh [--clean] [--small] [--cores N]

PLOT_TYPES=("private_v_nonprivate" "targeting" "epsilon" "engagement")
SCRIPT="distinguishing_game.py"

CLEAN_FLAG=""
SMALL_FLAG=""
CORES=8

for arg in "$@"
do
    if [ "$arg" == "--clean" ]; then
        CLEAN_FLAG="--clean"
    fi
    if [ "$arg" == "--small" ]; then
        SMALL_FLAG="--trials 100"
    fi
    if [[ "$arg" == --cores* ]]; then
        CORES="${arg#*=}"
        # Support both --cores=4 and --cores 4
        if [ "$CORES" == "--cores" ]; then
            shift
            CORES="$1"
        fi
    fi
done

for plot_type in "${PLOT_TYPES[@]}"
do
    echo "Running plot_type: $plot_type $CLEAN_FLAG $SMALL_FLAG --cores $CORES"
    python $SCRIPT --plot_type $plot_type $CLEAN_FLAG $SMALL_FLAG --cores $CORES
done