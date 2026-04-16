#!/bin/bash

# 1. Capture all input arguments
ALL_ARGS="$@"

# 2. Extract experiment name just for terminal display
args_array=("$@")
EXP_NAME="default_exp"
for ((i=0; i<${#args_array[@]}; i++)); do
    case ${args_array[$i]} in
        -n|--name) EXP_NAME="${args_array[$((i+1))]}" ;;
    esac
done

# 3. Critical Environment Variables for terminal rendering
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
export RICH_CONSOLE_WIDTH=400

echo "=========================================================="
echo "Submitting Task to TSP (Terminal Output Only)"
echo "Experiment: $EXP_NAME"
echo "=========================================================="

# 4. Execute training via tsp directly without any file redirection
tsp bash -c "python train.py $ALL_ARGS"