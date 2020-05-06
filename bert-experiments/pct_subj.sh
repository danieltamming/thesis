#!/bin/bash

set -e
gpu=0
aug_mode="trans"

for a in $(seq 0 6 24)
do
	(( b = a + 6 ))
	for pct_usage in $(seq 0.2 0.2 1.0)
	do
		python bert-experiments/pct_bert_subj.py -a $a -b $b -g $gpu -p $pct_usage
		for geo in $(seq 0.1 0.2 0.9)
		do
			python bert-experiments/pct_bert_subj.py -a $a -b $b -g $gpu -p $pct_usage -m $aug_mode -q $geo
		done
	done
done
