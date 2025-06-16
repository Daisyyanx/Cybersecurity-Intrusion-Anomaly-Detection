#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run IFOR model
python -m src.train \
    --train --test --seed 1 \
    --dataset beth \
    --model ifor \
    --use-wandb --wandb-name "ifor" --wandb-tags "ifor"