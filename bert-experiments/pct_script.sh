#!/bin/bash

set -e
gpu=0
a=24
b=30

for pct_usage in $(seq 0.2 0.2 0.8)
do
	python bert-experiments/pct_bert_trans_sst.py -a $a -b $b -g $gpu -p $pct_usage
	for geo in $(seq 0.1 0.2 0.9)
	do
		python bert-experiments/pct_bert_trans_sst.py -a $a -b $b -g $gpu -p $pct_usage -q $geo
	done
done
