# ../build/gpu -s 103 -o gpu.out -n 1000
# ./serial -s 103 -o serial.out -n 1000
# python correctness-check.py gpu.out serial.out

for seed in 10
do
    for par_num in 1000 2000 4000 10000 20000 40000
    do
        ../build/gpu -s $seed -n $par_num >> ./out/Test2.256.out
    done
done