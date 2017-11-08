#!/bin/bash

if [ -z $1 ]
  then
    echo "Please enter parameter(s) !"
    echo "The first parameter you enter after the executable is the number of processors you want to use."
    echo "The next parameters are optional :"
    echo "    -b  x_min  x_max  y_min  y_max"
    echo "    -d  larg  haut"
    echo "    -n  nb_iter"

else
  mpicc mandel.c -o mandel
  echo "mpicc mandel.c -o mandel"
  let "a = $1"
  shift
  mpiexec -machinefile hosts -n $a ./mandel $@
  echo "mpiexec -machinefile hosts -n $a ./mandel $@"
  # ajoute tous les fichiers mandel*.ppm dans un fichier .png les uns après les autres, nécessite le paquet "imagemagick"
  convert -append mandel*.ppm final.png
  echo "The final image of the Mandelbrot fractal was created in the file final.png"
  rm mandel*.ppm

fi
