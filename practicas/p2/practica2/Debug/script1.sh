#!/bin/bash

./practica2 -t $1 -T $2 -l 1 -h 4 -f 1 -s>$3.txt

./practica2 -t $1 -T $2 -l 1 -h 8 -f 1 -s>>$3.txt

./practica2 -t $1 -T $2 -l 1 -h 16 -f 1 -s>>$3.txt

./practica2 -t $1 -T $2 -l 1 -h 64 -f 1 -s>>$3.txt

./practica2 -t $1 -T $2 -l 2 -h 4 -f 1 -s>>$3.txt

./practica2 -t $1 -T $2 -l 2 -h 8 -f 1 -s>>$3.txt

./practica2 -t $1 -T $2 -l 2 -h 16 -f 1 -s>>$3.txt

./practica2 -t $1 -T $2 -l 2 -h 64 -f 1 -s>>$3.txt
