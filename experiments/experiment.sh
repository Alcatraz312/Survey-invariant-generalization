#!/bin/bash
cd /home/arbiter/projects/Survey-invariant-generalization

seeds=(123 42 85 156 234 301 412 500 617 724 836 901 999 1042)

for seed in "${seeds[@]}"; do
    echo "================================================"
    echo "Starting seed $seed at $(date)"
    echo "================================================"
    
    # run experiment
    python -m experiments.run_experiment --seed $seed
    
    # check if it succeeded
    if [ $? -eq 0 ]; then
        echo "Seed $seed completed successfully"
    else
        echo "Seed $seed FAILED — check logs"
    fi
    
    # force memory cleanup between runs
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 20   # give OS time to release memory
    
    echo ""
done

echo "All seeds completed at $(date)"