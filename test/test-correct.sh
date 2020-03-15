../build/gpu -s 103 -o gpu.out -n 1000
./serial -s 103 -o serial.out -n 1000
python correctness-check.py gpu.out serial.out
