echo "Regularización L1"
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1 >$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-1 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-2 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-3 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-4 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-5 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-6 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-7 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-8 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-9 >>$3
python3.7 rbf.py -t $1 -T $2 -c -r 0.05 -e 1e-10 >>$3

echo "Regularización L2"
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1 >$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-1 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-2 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-3 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-4 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-5 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-6 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-7 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-8 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-9 >>$3
python3.7 rbf.py -t $1 -T $2 -c -l -r 0.05 -e 1e-10 >>$3
