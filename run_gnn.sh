#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run GNN model
python -m src.train \
    --train --test --seed 1 \
    --dataset beth --subsample 20 \
    --model gnn \
    --use-wandb --wandb-name "gnn_subsample_20" --wandb-tags "gnn"
    # --epochs 20 \
    # --gnn-epochs 100 --learning-rate 0.001 \
    # --hidden-size 64 --num-layers 3 --k 5 \