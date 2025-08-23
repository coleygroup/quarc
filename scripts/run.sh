#!bin/bash

##################
#    Training    #
##################

## GNN Stage 1
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

## GNN Stage 2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 2 \
    --model-type gnn \
    --graph-hidden-size 768 \
    --depth 6 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage2_gnn \
    --output-size 32 \
    --num-classes 1376

## GNN Stage 3
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 3 \
    --model-type gnn \
    --graph-hidden-size 512 \
    --depth 3 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage3_gnn \
    --output-size 15 \
    --num-classes 1376

## GNN Stage 4
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 4 \
    --model-type gnn \
    --graph-hidden-size 512 \
    --depth 4 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage4_gnn \
    --output-size 27 \
    --num-classes 1376 \

## FFN Stage 1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 1 \
    --model-type ffn \
    --fp-dim 2048 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --max-epochs 2 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage1_ffn \
    --output-size 1376 \
    --num-classes 1376 \
    --train-data-path data/train.csv \
    --val-data-path data/val.csv \


## FFN Stage 2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 2 \
    --model-type ffn \
    --fp-dim 2048 \
    --hidden-size 2048 \
    --n-blocks 6 \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage2_ffn \
    --output-size 32 \
    --num-classes 1376

## FFN Stage 3
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 3 \
    --model-type ffn \
    --fp-dim 2048 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --activation "ReLU" \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage3_ffn \
    --output-size 15 \
    --num-classes 1376

## FFN Stage 4
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --stage 4 \
    --model-type ffn \
    --fp-dim 2048 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --activation "ReLU" \
    --max-epochs 30 \
    --batch-size 512 \
    --max-lr 1e-3 \
    --logger-name stage4_ffn \
    --output-size 27 \
    --num-classes 1376