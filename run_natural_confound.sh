#!/usr/bin/env bash

##############################################
# EXP 2: bird dataset with NATURAL confound
##############################################


# ==========================================
# ProtoPNet (LOWER BOUND)
# ==========================================

cmd_lower="python main.py experiment_name=\"lowerbound\" +experiment=natural_base_upper_lower"
echo $cmd_lower && $cmd_lower


# ==========================================
# ProtoPNet clean (UPPER BOUND)
# ==========================================

cmd_upper="python main.py experiment_name=\"upperbound\" +experiment=natural_upper"
echo $cmd_upper && $cmd_upper


# ==========================================
# IAIA-BL
# ==========================================
cmd_iaia="python main.py experiment_name=\"iaia\" +experiment=natural_iaiabl"
echo $cmd_iaia && $cmd_iaia


date
echo "Done!"