#!/bin/bash

SCRIPT_NAME="/home/evox5090ia/experiments/detection/train_veolia.py"
NEXT_SCRIPT="/home/evox5090ia/experiments/detection/train.py"
PYTHON="/home/evox5090ia/miniconda3/envs/py310_cuda129/bin/python"
TIME=300

echo "=== Waiting for '$SCRIPT_NAME' to finish... ==="
echo "Started at: $(date)"

while pgrep -f "$SCRIPT_NAME" > /dev/null 2>&1; do
    echo "Still running... $(date '+%H:%M:%S') - checking again in $TIME seconds"
    sleep $TIME
done

echo ""
echo "=== $SCRIPT_NAME run finished! Starting $NEXT_SCRIPT run... ==="
echo "Started at: $(date)"

$PYTHON "$NEXT_SCRIPT"

if [ $? -ne 0 ]; then
    echo "=== ERROR: $NEXT_SCRIPT failed! ==="
    exit 1
fi

echo ""
echo "=== $NEXT_SCRIPT run finished! ==="
echo "Finished at: $(date)"