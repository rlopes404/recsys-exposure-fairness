#!/bin/bash

dataset=$1

python3 main.py --train_filename "$dataset"-train.csv --valid_filename "$dataset"-valid.csv --test_filename "$dataset"-test.csv --train_mode True 1>>saida.log 2>>erro.log &       