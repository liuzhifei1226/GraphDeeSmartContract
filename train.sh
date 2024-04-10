#!/usr/bin/env bash
#for i in $(seq 1 5);
#do seed=$(( ( RANDOM % 10000 )  + 1 ));
python3 SMVulDetector.py --dataset REENTRANCY_CORENODES_1671 --model gcn_modify  --epochs 200 > GCN_modify_log/smartcheck_re.log;
#done

