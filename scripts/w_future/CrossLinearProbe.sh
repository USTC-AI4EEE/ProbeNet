#!/bin/bash
# ProbeNet 全量实验 — 并发调度（GPU 已在子脚本中分配）

SCRIPT_DIR="scripts/w_future/Probe/CrossLinear"

TASKS=(
    "NP_360.sh"
    "PJM_360.sh"
    "BE_360.sh"
    "FR_360.sh"
    "DE_360.sh"
    "Energy_360.sh"
    "Sdwpfm1_360.sh"
    "Sdwpfm2_360.sh"
    "Sdwpfh1_360.sh"
    "Sdwpfh2_360.sh"
    "Colbun_30.sh"
    "Rapel_30.sh"
)

echo "=== CrossLinear benchmark: ${#TASKS[@]} tasks ==="

for script in "${TASKS[@]}"; do
    log="logs/${script%.sh}.log"
    mkdir -p logs
    echo "  start: $script -> $log"
    bash "$SCRIPT_DIR/$script" &> "$log" &
done

echo "All tasks dispatched. Waiting..."
wait
echo "=== All done ==="
