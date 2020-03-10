import os
import random
import pickle
from collections import Counter

import numpy as np

def partition_within_classes(data, pct_in_A, make_A_balanced, seed=0):
	'''
	Partitions data into A and B, where A pct_in_A of the dataset and B is
	the rest. A will have perfectly even distribution among classes. 
	If the dataset is imbalanced then B will be imbalanced

	TODO: SET SEED AND USE RANDOM. CURRENTLY DETERMINISTIC BUT WIHTOUT SEED
	'''
	random.seed(seed)
	random.shuffle(data)
	if pct_in_A == 1:
		return data, []
	num_classes = len(set([tup[0] for tup in data]))

	print(len([1 for tup in data if tup[0] == 0]), 
		  len([1 for tup in data if tup[0] == 1]))

	if make_A_balanced:
		A_size_per_class = int(pct_in_A*len(data)/num_classes)
		A_class_deficits = {i:-A_size_per_class for i in range(num_classes)}
		A, B = [], []
		for i, tup in enumerate(data):
			label = tup[0]
			if A_class_deficits[label] < 0:
				A.append(tup)
				A_class_deficits[label] += 1
			else:
				B.append(tup)
		print(len([1 for tup in A if tup[0] == 0]), 
			  len([1 for tup in A if tup[0] == 1]))
		return A, B
	else:
		label_dict = {}
		for tup in data:
			label = tup[0]
			if label in label_dict:
				label_dict[label].append(tup)
			else:
				label_dict[label] = [tup]
		A, B = [], []
		for arr in label_dict.values():
			A_label_size = int(pct_in_A*len(arr))
			A.extend(arr[:A_label_size])
			B.extend(arr[A_label_size:])
		return A, B

def read_no_aug(set_path, input_length, is_bytes):
	if is_bytes:
		read_code = 'rb'
	else:
		read_code = 'r'
	set_data = []
	with open(set_path, read_code) as f:
		for line in f.read().splitlines():
			if is_bytes:
				line = line.decode('latin-1')
			label, example = line.split(maxsplit=1)
			label = int(label)
			example = example.replace('-lrb-', '(').replace('-rrb-', ')').replace('\\/', ' / ')
			if input_length is not None:
				example = ' '.join(example.split()[:input_length])
			else:
				example = ' '.join(example.split())
			set_data.append((label, example, None))
	return set_data

def read_trans_aug(set_path):
	# not restricting to input length here but it isn't that necessary anyways
	# that offers more time savings in synonym as it restricts the search
	set_data = []
	with open(set_path) as f:
		line = f.readline().strip('\n')
		while line:
			label = int(line)
			example = f.readline().strip('\n')
			aug_counter = Counter()
			line = f.readline().strip('\n')
			while line:
				count, aug_example = line.split(maxsplit=1)
				count = int(count)
				aug_counter[aug_example] = count
				line = f.readline().strip('\n')
			set_data.append((label, example, aug_counter))
			line = f.readline().strip('\n')
	return set_data

def read_context_aug(aug_data_path, pct_usage, small_label, 
					 small_prop, seed):
	'''
	Unlike trans aug, this only retrieves that train set
	'''
	filename = '{}-{}-{}-{}.pickle'.format(pct_usage, small_label, 
											small_prop, seed)
	filepath = os.path.join(aug_data_path, filename)
	with open(filepath, 'rb') as f:
		data = pickle.load(f)
	return data

def get_sst(input_length, aug_mode, pct_usage=None, 
			small_label=None, small_prop=None, seed=None):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	data_path = os.path.join(data_parent,'sst')
	data_dict = {}
	if aug_mode is None or aug_mode == 'synonym':
		for set_name in ['train', 'dev', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			data_dict[set_name] = read_no_aug(
				set_path, input_length, False)
		return data_dict
	elif aug_mode == 'trans':
		aug_data_path = os.path.join(data_path,'trans_aug')
		for set_name in ['train', 'dev', 'test']:
			set_path = os.path.join(aug_data_path, set_name+'.txt')
			data_dict[set_name] = read_trans_aug(set_path)
		return data_dict
	elif aug_mode == 'context':
		aug_data_path = os.path.join(data_path, 'context_aug')
		for set_name in ['dev', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			data_dict[set_name] = read_no_aug(
				set_path, input_length, False)
		data_dict['train'] = read_context_aug(
			aug_data_path, pct_usage, small_label, small_prop, seed)
		return data_dict
	else:
		raise ValueError('Unrecognized augmentation.')

def get_subj(input_length, aug_mode, seed=None):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	data_path = os.path.join(data_parent, 'subj')
	if aug_mode is None or aug_mode == 'synonym':
		file_path = os.path.join(data_path,'subj.txt')
		all_data = read_no_aug(file_path, input_length, True)
		# let 10% of data be the development set
		# dev_data, train_data = partition_within_classes(all_data, 0.1, False)
		# return {'dev': dev_data, 'train': train_data}
	elif aug_mode == 'trans':
		aug_file_path = os.path.join(data_path, 'trans_aug/subj.txt')
		all_data = read_trans_aug(aug_file_path)
	elif aug_mode == 'context':
		context_train_data = read_context_aug(
			data_path+'/context_aug/train.pickle')
	else:
		raise ValueError('Unrecognized augmentation.')
	dev_data, train_data = partition_within_classes(all_data, 0.1, False)
	if aug_mode == 'context':
		train_data = context_train_data
	return {'dev': dev_data, 'train': train_data}

def get_trec(input_length, aug_mode):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	target_parent = os.path.join(repo_path, 'data')
	data_path = os.path.join(data_parent,'trec')
	data_dict = {}
	if aug_mode is None or aug_mode == 'synonym':
		for set_name in ['train', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			data_dict[set_name] = read_no_aug(set_path, input_length, True)
	elif aug_mode == 'trans':
		aug_data_path = os.path.join(data_path, 'trans_aug')
		for set_name in ['train', 'test']:
			set_path = os.path.join(aug_data_path, set_name+'.txt')
			data_dict[set_name] = read_trans_aug(set_path)
	else:
		raise ValueError('Unrecognized augmentation.')
	# split given training split into training and dev set
	dev_data, train_data = partition_within_classes(data_dict['train'], 0.1, False)
	data_dict['dev'] = dev_data
	data_dict['train'] = train_data
	return data_dict