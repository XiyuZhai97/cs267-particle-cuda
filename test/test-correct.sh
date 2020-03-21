../build/gpu -s 103 -o gpu.out -n 4
./serial -s 103 -o serial.out -n 4
python correctness-check.py gpu.out serial.out
