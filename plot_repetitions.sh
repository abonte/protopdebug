#!/usr/bin/env bash

find $1 -maxdepth 1 -mindepth 1 -type d | while read line; do
  echo $line
  python plot_stat.py -o $line comparison -modeldir $line
  #python visualization/plot_prototypes.py -modeldir $line

  #modelpath=$(find $line -maxdepth 1 -type f -name "*nopush*.pth.tar" | tail -n 1)
  #echo $modelpath
  #python global_analysis.py $modelpath
done
