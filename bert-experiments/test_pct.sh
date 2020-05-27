#!/bin/bash

set -e
gpu=0
# aug_mode="synonym"

# script="bert-experiments/test_bert_sst.py"
# script="bert-experiments/test_bert_subj.py"
script="bert-experiments/test_bert_sfu.py"

for aug_mode in "synonym" "trans"
do
	for a in $(seq 0 6 24)
	do
		(( b = a + 6 ))
		for pct_usage in $(seq 0.2 0.2 1.0)
		do
			python $script -a $a -b $b -g $gpu -m $aug_mode -p $pct_usage
			python $script -a $a -b $b -g $gpu -m $aug_mode -p $pct_usage -o
		done
	done
done

aug_mode="context"
for a in 0 5
do
	(( b = a + 5 ))
	for pct_usage in $(seq 0.2 0.2 1.0)
	do
		python $script -a $a -b $b -g $gpu -m $aug_mode -p $pct_usage
		python $script -a $a -b $b -g $gpu -m $aug_mode -p $pct_usage -o
	done
done