import os
import random
import pickle

import numpy as np

def partition_within_classes(data, pct_in_A, make_A_balanced):
	'''
	Partitions data into A and B, where A pct_in_A of the dataset and B is
	the rest. A will have perfectly even distribution among classes. 
	If the dataset is imbalanced then B will be imbalanced

	TODO: SET SEED AND USE RANDOM. CURRENTLY DETERMINISTIC BUT WIHTOUT SEED
	'''
	if pct_in_A == 1:
		return data, []
	num_classes = len(set([label for label, _ in data]))

	if make_A_balanced:
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
	else:
		label_dict = {}
		for label, example in data:
			if label in label_dict:
				label_dict[label].append((label, example))
			else:
				label_dict[label] = [(label, example)]
		A, B = [], []
		for arr in label_dict.values():
			A_label_size = int(pct_in_A*len(arr))
			A.extend(arr[:A_label_size])
			B.extend(arr[A_label_size:])
		return A, B