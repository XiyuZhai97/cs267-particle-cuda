#!/bin/bash
rm ./out/Test3.256.out
sbatch job-gpu-log
# sbatch job-gpu-naive-log
# for seed in 10
# do
#     for par_num in 1000 2000 4000 10000 20000 40000
#     do
#         ../build/gpu -s $seed -n $par_num >> ./out/Test2.256.out
#     done
# done