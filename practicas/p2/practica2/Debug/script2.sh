#!/bin/bash

echo "Script para vote iniciado"
./script1.sh /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat vote
echo "Script para vote terminado"

echo "Script para nomnist iniciado"
./script1.sh /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat nomnist
echo "Script para nomnist terminado"
