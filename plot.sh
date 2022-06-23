#!/usr/bin/env bash

model=$(echo $1 | rev | cut -d '/' -f 1 | rev)
modeldir=$(echo $1 | rev | cut -d '/' -f 2- | rev)

classes=$2

# classes="0 8 14 6 15" # Cub200
# classes="0 1" # COVID

python plot_stat.py -o $modeldir comparison -modeldir $modeldir 
python visualization/plot_prototypes.py -modeldir $modeldir -classes $classes
python global_analysis.py $1 && \
python visualization/plot_global_analysis.py $1 -classes $classes
