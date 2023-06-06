#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 ./leaderboard/scripts/run_evaluation_baseline_latefusion_28p_to_100p.sh
#CUDA_VISIBLE_DEVICES=1 ./leaderboard/scripts/run_evaluation_baseline_latefusion_28p_to_100p-V2.sh

cmds=(
./leaderboard/scripts/run_evaluation_latefusion_res50.sh
./leaderboard/scripts/run_evaluation_transfuser_res50.sh
./leaderboard/scripts/run_evaluation_SAC_5vs3.sh
./leaderboard/scripts/run_evaluation_SAC_4vs4.sh
)

for cmd in ${cmds[@]}
do
  echo ${cmd}" start!"
  CUDA_VISIBLE_DEVICES=1 ${cmd}
  echo ${cmd}" finished!"
done


