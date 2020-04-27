#!/bin/bash

set -e
gpu=0
# a=3
# b=4
# pct_usage=0.1

# for pct_usage in $(seq 0.1 0.1 1.0)
# do
# 	python augs/create_context.py -a $a -b $b -g $gpu -u $pct_usage
# done

for pct_usage in $(seq 0.3 0.1 0.6)
do
	for split_num in {0..10}
	do
		python augs/create_context.py -g $gpu -u $pct_usage -s $split_num
	done
done