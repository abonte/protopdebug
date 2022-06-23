#!/usr/bin/env bash

##############################################
# EXP 1: bird dataset with ARTIFICIAL confound
##############################################

repetition="seed=range(5,20)"


cmd_aggregation="python main.py --multirun $repetition
                     experiment_name=forgetting_loss
                     +experiment=artificial_aggregation"
echo $cmd_aggregation && $cmd_aggregation




cmd_lower="python main.py --multirun $repetition
                    experiment_name=lowerbound
                    +experiment=artificial_lower"
echo $cmd_lower && $cmd_lower




cmd_upper="python main.py --multirun $repetition
                    experiment_name=upperbound
                    +experiment=artificial_upper"
echo $cmd_upper && $cmd_upper




cmd_iaiabl="python main.py --multirun $repetition
                           experiment_name=iaiabl
                           +experiment=artificial_iaiabl
                           debug.fine_annotation=3
                           model.coefs.debug=0.01"
echo $cmd_iaiabl && $cmd_iaiabl



cmd_iaiabl="python main.py --multirun $repetition
                           experiment_name=iaiabl
                           +experiment=artificial_iaiabl
                           debug.fine_annotation=0.05"
echo $cmd_iaiabl && $cmd_iaiabl


cmd_iaiabl="python main.py --multirun $repetition
                           experiment_name=iaiabl
                           +experiment=artificial_iaiabl
                           debug.fine_annotation=0.2"
echo $cmd_iaiabl && $cmd_iaiabl


cmd_iaiabl="python main.py --multirun $repetition
                           experiment_name=iaiabl
                           +experiment=artificial_iaiabl
                           debug.fine_annotation=1.0"
echo $cmd_iaiabl && $cmd_iaiabl


date
echo 'Done!'
