#!/bin/bash

./practica1 -t $1 -T $2 -l 1 -h 2 >$3.txt

./practica1 -t $1 -T $2 -l 1 -h 4 >>$3.txt

./practica1 -t $1 -T $2 -l 1 -h 8 >>$3.txt

./practica1 -t $1 -T $2 -l 1 -h 32 >>$3.txt

./practica1 -t $1 -T $2 -l 1 -h 64 >>$3.txt

./practica1 -t $1 -T $2 -l 1 -h 100 >>$3.txt

./practica1 -t $1 -T $2 -l 2 -h 2 >>$3.txt

./practica1 -t $1 -T $2 -l 2 -h 4 >>$3.txt

./practica1 -t $1 -T $2 -l 2 -h 8 >>$3.txt

./practica1 -t $1 -T $2 -l 2 -h 32 >>$3.txt

./practica1 -t $1 -T $2 -l 2 -h 64 >>$3.txt

./practica1 -t $1 -T $2 -l 2 -h 100 >>$3.txt


