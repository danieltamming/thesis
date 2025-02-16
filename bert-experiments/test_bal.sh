#!/bin/bash

set -e
gpu=0

# script="bert-experiments/test_bert_sst.py"
# script="bert-experiments/test_bert_subj.py"
script="bert-experiments/test_bert_sfu.py"


# for aug_mode in "synonym" "trans"
# do
# 	for a in $(seq 0 2 28)
# 	do
# 		(( b = a + 2 ))
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

aug_mode="context"
for a in $(seq 0 2 8)
do
	(( b = a + 2 ))
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