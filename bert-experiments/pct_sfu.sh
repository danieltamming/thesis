#!/bin/bash

set -e
gpu=1
aug_mode="syn"

# for a in $(seq 0 6 24)
# do
# 	(( b = a + 6 ))
# 	for pct_usage in $(seq 0.2 0.2 1.0)
# 	do
# 		python bert-experiments/pct_bert_sfu.py -a $a -b $b -g $gpu -p $pct_usage -m $aug_mode
# 		for geo in $(seq 0.1 0.2 0.9)
# 		do
# 			python bert-experiments/pct_bert_sfu.py -a $a -b $b -g $gpu -p $pct_usage -m $aug_mode -q $geo
# 		done
# 	done
# done

a=0
b=2
pct_usage=1.0

for learning_rate in 0.000005 0.000002 0.000001
do
	python bert-experiments/pct_bert_sfu.py -a $a -b $b -g $gpu -p $pct_usage -m $aug_mode -l $learning_rate
	# echo $learning_rate
done
