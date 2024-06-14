#!/usr/bin/env bash
#for i in $(seq 1 5);
#do seed=$(( ( RANDOM % 10000 )  + 1 ));
nohup python3 SMVulDetector.py --dataset REENTRANCY_FULLNODES_1671 --model gcn_origin --dropout 0.2 --lr 0.001 --seed 50 --epochs 300 -b 128 > train_log/GCN_origin/re.log 2>&1 &
nohup python3 SMVulDetector.py --dataset REENTRANCY_CORENODES_1671 --model graphsage --epochs 200 > train_log/graphsage/2024.6.13.log 2>&1 &

#done

