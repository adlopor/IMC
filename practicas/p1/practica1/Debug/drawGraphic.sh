#!/bin/bash

cat << _end_ | gnuplot
set terminal postscript eps color
set output "$2"
set key right top
set xlabel "Numero de iteracion del algoritmo"
set ylabel "Valor de error"
plot "$1" using 1 t "Error entrenamiento"  w l,"$1" using 2 t "Error test" w l,"$1" using 3 t "Error validacion" w l
_end_
