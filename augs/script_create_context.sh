#!/bin/bash

set -e
gpu=0
a=7
b=8

for pct_usage in $(seq 0.2 0.1 1.0)
do
	python augs/create_context.py -a $a -b $b -g $gpu -u $pct_usage
done
