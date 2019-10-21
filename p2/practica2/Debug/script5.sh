#!/bin/bash

echo "Script para vote iniciado"
./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -f 1 -o -s >vote05.csv
echo "Script para vote terminado"

echo "Script para nomnist iniciado"
./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -f 1 -o -s >nomnist05.csv
echo "Script para nomnist terminado"
