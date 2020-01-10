#!/bin/bash

./script3.sh /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/train_xor.dat /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/test_xor.dat xor_d_v 2 100
./script3.sh /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/train_sin.dat /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/test_sin.dat sin_d_v 2 64
./script3.sh /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/train_quake.dat /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/test_quake.dat quake_d_v 1 32
./script3.sh /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/train_parkinsons.dat /home/adrian/Escritorio/IMC/basesDatosPr1IMC/dat/test_parkinsons.dat parkinsons_d_v 2 64
