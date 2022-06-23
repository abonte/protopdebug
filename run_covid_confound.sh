#!/usr/bin/env bash

##############################################
# EXP 3: COVID dataset
##############################################


cmd="python main.py experiment_name=first_round +experiment=covid_base"
$cmd

#debug 200
#rem 10
# clst 0.1
# sep 0.001
# crs_ent 0.4

cmd="python main.py experiment_name=\"second_round\"
                    +experiment=covid_aggregation
                    debug.path_to_model='$1'"
$cmd

date
echo 'Done!'
