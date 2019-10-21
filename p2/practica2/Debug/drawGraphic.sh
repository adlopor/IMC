#!/bin/bash

cat << _end_ | gnuplot
set terminal postscript eps color
set output "$2"
set key right top
set xlabel "Numero de iteracion del algoritmo"
set ylabel "Porcentaje de CCR"
plot "$1" using 1 t "CCR entrenamiento"  w l,"$1" using 2 t "CCR test" w l
_end_
