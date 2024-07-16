#!/usr/bin/env bash
#for i in $(seq 1 5);
#do seed=$(( ( RANDOM % 10000 )  + 1 ));
nohup python3 SMVulDetector.py --dataset REENTRANCY_FULLNODES_1671 --model gcn_modify --dropout 0.3 --lr 0.01 --seed 50 --epochs 200 -b 128 > train_log/GCN_modify_log/re.log 2>&1 &
nohup python3 SMVulDetector.py --dataset REENTRANCY_CORENODES_1671 --model graphsage --epochs 200 > train_log/graphsage/2024.6.13.log 2>&1 &
nohup ./smartbugs -t Oyente -f dataset/vul_kinds_marked/*.sol --processes 6 --mem-limit 8g --timeout 600 > output.txt 2>&1 &

#done

