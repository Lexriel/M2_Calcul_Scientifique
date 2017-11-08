#!/bin/bash

# The script takes as argument the name of the file
#$1 : file name

sed -i '/>/d' $1
#rm $1

