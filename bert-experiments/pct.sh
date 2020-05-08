#!/bin/bash

set -e
gpu=1
aug_mode="context"

script="bert-experiments/bert_sst.py"
# script="bert-experiments/bert_subj.py"
# script="bert-experiments/bert_sfu.py"

for a in 0 5
do
	(( b = a + 5 ))
	for pct_usage in $(seq 0.2 0.2 1.0)
	do
		python $script -a $a -b $b -g $gpu -p $pct_usage -m $aug_mode
		for geo in $(seq 0.1 0.2 0.9)
		do
			python $script -a $a -b $b -g $gpu -p $pct_usage -m $aug_mode -q $geo
		done
	done
done