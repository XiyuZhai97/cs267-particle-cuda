# ../build/gpu -s 103 -o gpu.out -n 4
# ./serial -s 103 -o serial.out -n 4
sbatch ./job-gpu-m-naive
python correctness-check.py ./gpu_1000.out ./gpu_naive_1000.out
