python3.7 rbf.py -t $1 -T $2 -r 0.05 -e 1e-5 -o 2 >$3
python3.7 rbf.py -t $1 -T $2 -r 0.15 -e 1e-5 -o 2 >>$3
python3.7 rbf.py -t $1 -T $2 -r 0.25 -e 1e-5 -o 2 >>$3
python3.7 rbf.py -t $1 -T $2 -r 0.5 -e 1e-5 -o 2 >>$3
