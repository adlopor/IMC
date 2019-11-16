python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-5 -o 1 >$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.15 -e 1e-5 -o 1 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.25 -e 1e-5 -o 1 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.5 -e 1e-5 -o 1 >>$3
