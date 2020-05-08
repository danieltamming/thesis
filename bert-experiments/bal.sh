#!/bin/bash

set -e
gpu=0
aug_mode="synonym"

script="bert-experiments/bert_sst.py"
# script="bert-experiments/bert_subj.py"
# script="bert-experiments/bert_sfu.py"

for a in $(seq 0 6 24)
do
	(( b = a + 6 ))
	for small_prop in $(seq 0.2 0.2 0.8)
	do
		for small_label in 0 1
		do
			for undersample in 0 1
			do
				python $script -a $a -b $b -g $gpu -m $aug_mode -r $small_prop -l $small_label -u $undersample
			done
			for geo in $(seq 0.1 0.2 0.9)
			do
				python $script -a $a -b $b -g $gpu -m $aug_mode -r $small_prop -l $small_label -q $geo
			done
		done
	done
done