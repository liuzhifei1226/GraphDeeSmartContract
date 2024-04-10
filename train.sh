#!/usr/bin/env bash
#for i in $(seq 1 5);
#do seed=$(( ( RANDOM % 10000 )  + 1 ));
python3 SMVulDetector.py --dataset REENTRANCY_CORENODES_1671 --model gat  --epochs 200 > GAT_log/smartcheck_re.log;
#done

