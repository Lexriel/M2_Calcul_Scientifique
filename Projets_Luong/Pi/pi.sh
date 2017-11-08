#!/bin/bash

if [ -z $2 ]
  then
    echo "Please enter two parameters after the name of the executable !"
    echo "First parameter : number of processors 'P' you want to use, it must be a divisor of 2*10^(precision+1), otherwise results will be false."
    echo "Second parameter : number of decimals 'precision' you want to be true after the point in the expression of pi."

else
  mpicc pi.c -o ./pi
  echo "mpicc pi.c -o ./pi"
  echo "mpiexec -machinefile hosts -n $1 ./pi $2"
  mpiexec -machinefile hosts -n $1 ./pi $2
  echo "The approximation of pi is saved in the file pi.dat."
fi
