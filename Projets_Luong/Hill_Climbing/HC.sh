#!/bin/bash

if [ -z $1 ]
  then
    echo "Please enter one parameter !"
    echo "The parameter you enter after the executable is the number of iterations you want to use."

else
  let "a = $1"
  shift
  nvcc part4.cu -o part4
  echo "nvcc part4.cu -o part4"
  list= `ls tai*.dat`

  for files in list
  do
    part4 $1
    echo "part4 $1"
    shift
  done
