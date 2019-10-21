#!/bin/bash

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -v 0 -d 1 -f 1 -s >offlinevote.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -v 0.15 -d 1 -f 1 -s >>offlinevote.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -v 0.25 -d 1 -f 1 -s >>offlinevote.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -v 0 -d 2 -f 1 -s >>offlinevote.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -v 0.15 -d 2 -f 1 -s >>offlinevote.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_vote.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_vote.dat -l 1 -h 16 -v 0.25 -d 2 -f 1 -s >>offlinevote.csv
echo "vote off-line terminado"

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -v 0 -d 1 -f 1 -s >offlinenomnist.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -v 0.15 -d 1 -f 1 -s >>offlinenomnist.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -v 0.25 -d 1 -f 1 -s >>offlinenomnist.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -v 0 -d 2 -f 1 -s >>offlinenomnist.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -v 0.15 -d 2 -f 1 -s >>offlinenomnist.csv

./practica2 -t /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/train_nomnist.dat -T /home/adrian/Escritorio/IMC/p2/basesDatosPr2IMC/dat/test_nomnist.dat -l 1 -h 64 -v 0.25 -d 2 -f 1 -s >>offlinenomnist.csv
echo "noMNIST off-line terminado"
