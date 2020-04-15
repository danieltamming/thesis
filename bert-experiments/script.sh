#!/bin/bash

set -e
gpu=0
a=0
b=4

for small_prop in $(seq 0.1 0.1 0.9)
do
	for small_label in 0 1
	do
		for undersample in 0 1
		do
			python bert-experiments/bal_bert_trans_sst.py -a $a -b $b -g $gpu -p $small_prop -l $small_label -u $undersample
		done
		for geo in $(seq 0.1 0.1 0.9)
		do
			python bert-experiments/bal_bert_trans_sst.py -a $a -b $b -g $gpu -p $small_prop -l $small_label -q $geo
		done
	done
done