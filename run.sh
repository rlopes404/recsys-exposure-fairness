#!/bin/bash

#ratio_values=(0.9 0.7 0.5 0.3 0.1)
fairness=$1
dataset=$2
path=$(pwd)
ratio_values=(0.9 0.3)

for ratio in ${ratio_values[*]};
do
    python3 main.py --train_filename "$dataset"-train.csv --valid_filename "$dataset"-valid.csv --test_filename "$dataset"-test.csv --fairness_constraint $fairness --top_ratio $ratio  --is_knn True  1>>saida.log 2>>erro.log &       
done
