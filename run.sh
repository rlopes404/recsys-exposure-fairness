#!/bin/bash

ratio_values=(0.9 0.7 0.5 0.3 0.1)
fairness=$1
path=$(pwd)


for ratio in ${ratio_values[*]};
do
    python3 main.py --train_filename=ml1m_5_train.csv --valid_filename=ml1m_5_valid.csv --test_filename=ml1m_5_test.csv --fairness_constraint=$fairness --top_ratio=$ratio 1>saida.log 2>erro.log &       
done
