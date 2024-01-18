#!/bin/sh

#result_dir="solve_time_hidden_8_64_l1_51_grad_41_best.txt"
#cd "/home/jinxin/xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1/" || exit
#for idx in $(seq 180 205);
#do
#    julia --project='~/xjx/SRIBD/PowerModelsSecurityConstrained.jl' test_all.jl --start "$idx" --end "$idx" --result_dir "$result_dir" --second_stage
#done


result_dir="solve_time_hidden_8_128_8_l1_41_grad_41_best.txt"
cd "/home/jinxin/xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1/" || exit
for idx in $(seq 180 205);
do
  julia --project='~/xjx/SRIBD/PowerModelsSecurityConstrained.jl' test_all.jl --start "$idx" --end "$idx" --result_dir "$result_dir" --second_stage
done