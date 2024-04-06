#!/usr/bin/env bash
for i in $(seq 1 5);
do seed=$(( ( RANDOM % 10000 )  + 1 ));
python3 SMVulDetector.py --dataset LOOP_CORENODES_1317 --model gcn_modify --seed $seed --epochs 100 > log/smartcheck_loop_"$i".log;
done

