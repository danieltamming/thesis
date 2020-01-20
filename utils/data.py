import os
import random
import pickle

import numpy as np

def partition_within_classes(data, pct_in_A):
	'''
	Partitions data into A and B, where A pct_in_A of the dataset and B is
	the rest. A will have perfectly even distribution among classes. 
	If the dataset is imbalanced then B will be imbalanced

	TODO: SET SEED AND USE RANDOM. CURRENTLY DETERMINISTIC BUT WIHTOUT SEED
	'''
	if pct_in_A == 1:
		return data, []
	num_classes = len(set([label for label, _ in data]))

	print('Num classes: ', num_classes)

	A_size_per_class = int(pct_in_A*len(data)/num_classes)
	A_class_deficits = {i:-A_size_per_class for i in range(num_classes)}
	A, B = [], []
	for i, (label, example) in enumerate(data):
		if A_class_deficits[label] < 0:
			A.append((label, example))
			A_class_deficits[label] += 1
		else:
			B.append((label, example))
	return A, B