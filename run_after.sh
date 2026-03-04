#!/bin/bash

SCRIPT_NAME="sam3_gencuts_route.py"
ESQUERDA_SCRIPT="/home/evox5090ia/sam3_gencuts_route_esquerda.py"

echo "=== Waiting for '$SCRIPT_NAME' (direita) to finish... ==="
echo "Started at: $(date)"

while pgrep -f "$SCRIPT_NAME" > /dev/null 2>&1; do
    echo "  Still running... $(date '+%H:%M:%S') - checking again in 60s"
    sleep 300
done

echo ""
echo "=== direita run finished! Starting esquerda run... ==="
echo "Started at: $(date)"

python "$ESQUERDA_SCRIPT"

echo ""
echo "=== esquerda run finished! ==="
echo "Finished at: $(date)"