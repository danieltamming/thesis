#!/bin/bash

set -e
gpu=0
a=0
b=6

for pct_usage in $(seq 0.1 0.1 1.0)
do
	python augs/create_context.py -a $a -b $b -g $gpu -u $pct_usage
done