#!/bin/bash
CUDA_VISIBLE_DEVICES=1 ./leaderboard/scripts/run_evaluation_baseline_latefusion_28p_to_100p_ours.sh
CUDA_VISIBLE_DEVICES=1 ./leaderboard/scripts/run_evaluation_baseline_latefusion_28p_to_100p_ours-V2.sh
CUDA_VISIBLE_DEVICES=1 ./leaderboard/scripts/run_evaluation_baseline_latefusion_28p-V2.sh

