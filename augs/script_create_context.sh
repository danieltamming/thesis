#!/bin/bash

set -e
gpu=1
a=7
b=8
# pct_usage=1.0

for pct_usage in $(seq 0.2 0.1 1.0)
do
	python augs/create_context.py -a $a -b $b -g $gpu -u $pct_usage
done

# for seed in {0..10}
# do
# 	python augs/create_context.py -g $gpu -u $pct_usage -s $seed
# done
