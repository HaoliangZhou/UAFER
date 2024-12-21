#!/bin/bash

# ========================================= rafdb =========================================
CLASS=7
DATASET="rafdb"
DATAPATH="/data/RAFDB/basic/"


python3 train.py \
--classes ${CLASS} \
--dataset ${DATASET} \
--data-path ${DATAPATH} \
--loss_function edl \
--alpha 1e-1 \
--lamda_kd 1 \
--lamda_ruc 1 \
--topk 16 \
--fix_layer 2 \
--batch-size 128 \
--save_model True \
--save_np True \
--gpu 6


# ========================================= affectnet =========================================
CLASS=7
DATASET="affectnet"  # "affectnet" or "affectnet_8"
DATAPATH="/data/AffectNet/"


python3 train_affectnet.py \
--classes ${CLASS} \
--dataset ${DATASET} \
--data-path ${DATAPATH} \
--loss_function edl \
--alpha 3e-1 \
--lamda_kd 1 \
--lamda_ruc 1 \
--topk 16 \
--fix_layer 8 \
--batch-size 256 \
--save_model True \
--save_np True \
--gpu 6