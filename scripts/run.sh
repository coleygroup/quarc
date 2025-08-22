#!bin/bash

# Run preprocessing
python scripts/preprocess.py \
    --config configs/preprocess_config.yaml \
    --all

# Run training for all stages (distributed used by default, use torchrun if using multiple gpus)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 1 \
    --model-type gnn \
    --graph-hidden-size 1024 \
    --depth 2 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage1_gnn \
    --output-size 1376 \
    --num-classes 1376

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 2 \
    --model-type ffn \
    --hidden-size 2048 \
    --n-blocks 3 \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage2_ffn \
    --output-size 1376 \
    --num-classes 1376