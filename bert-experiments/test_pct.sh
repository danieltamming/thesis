#!/bin/bash

set -e
gpu=0
aug_mode="context"

# script="bert-experiments/test_bert_sst.py"
# script="bert-experiments/test_bert_subj.py"
# script="bert-experiments/test_bert_sfu.py"

for script in "bert-experiments/test_bert_sst.py" "bert-experiments/test_bert_subj.py"
do
	for a in 0 5
	do
		(( b = a + 5 ))
		for pct_usage in $(seq 0.2 0.2 1.0)
		do
			python $script -a $a -b $b -g $gpu -m $aug_mode -p $pct_usage -o
			python $script -a $a -b $b -g $gpu -m $aug_mode -p $pct_usage
		done
	done
done
