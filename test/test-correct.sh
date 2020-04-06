# ../build/gpu -s 103 -o gpu.out -n 4
# ./serial -s 103 -o serial.out -n 4
sbatch ../build/job-gpu
python correctness-check.py gpu_1000.out serial_1000.out
