#!/bin/bash

echo "===== Stage 1: YOLO11s 50 epochs ====="

python train.py --data configs/gc10_det.yaml --model yolo11s.pt --epochs 50 --batch 8 --device 0 --name gc10_yolo11s_50e

echo "===== Stage 2: YOLO11m 50 epochs ====="

python train.py --data configs/gc10_det.yaml --model yolo11m.pt --epochs 50 --batch 4 --device 0 --name gc10_yolo11m_50e

echo "===== ALL TRAINING FINISHED ====="
