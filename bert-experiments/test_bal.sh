#!/bin/bash

set -e
gpu=1

# script="bert-experiments/test_bert_sst.py"
# script="bert-experiments/test_bert_subj.py"
script="bert-experiments/test_bert_sfu.py"


for aug_mode in "synonym" "trans"
do
	for a in $(seq 0 6 24)
	do
		(( b = a + 6 ))
		for small_prop in $(seq 0.2 0.2 0.8)
		do
			for small_label in 0 1
			do
				for undersample in 0 1
				do
					python $script -a $a -b $b -g $gpu -m $aug_mode -r $small_prop -l $small_label -u $undersample -o
				done
				python $script -a $a -b $b -g $gpu -m $aug_mode -r $small_prop -l $small_label
			done
		done
	done
done

# aug_mode="context"
# for script in "bert-experiments/test_bert_sst.py" "bert-experiments/test_bert_subj.py" "bert-experiments/test_bert_sfu.py"
# do
# 	for a in 0 5
# 	do
# 		(( b = a + 5 ))
# 		for small_prop in $(seq 0.2 0.2 0.8)
# 		do
# 			for small_label in 0 1
# 			do
# 				for undersample in 0 1
# 				do
# 					python $script -a $a -b $b -g $gpu -m $aug_mode -r $small_prop -l $small_label -u $undersample -o
# 				done
# 				python $script -a $a -b $b -g $gpu -m $aug_mode -r $small_prop -l $small_label
# 			done
# 		done
# 	done
# done