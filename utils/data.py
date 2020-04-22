import os
import random
import pickle
from collections import Counter

import numpy as np

def partition_within_classes(data, pct_in_A, make_A_balanced, 
							 seed=0, split_num=0):
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
			# A_label_size = int(pct_in_A*len(arr))
			# A.extend(arr[:A_label_size])
			# B.extend(arr[A_label_size:])
			A_label_count = int(pct_in_A*len(arr))
			A_start_idx = split_num*A_label_count
			A_end_idx = (split_num+1)*A_label_count
			A.extend(arr[A_start_idx:A_end_idx])
			B.extend(arr[:A_start_idx])
			B.extend(arr[A_end_idx:])
		print(len([1 for tup in A if tup[0] == 0]), 
			  len([1 for tup in A if tup[0] == 1]))
		return A, B

def read_no_aug(set_path, input_length, is_bytes, ignore_label):
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
			if label != ignore_label:
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
					 small_prop, seed, split_num=None):
	'''
	Unlike trans aug, this only retrieves that train set
	'''
	if pct_usage is None:
		pct_usage_display = pct_usage
	else:
		pct_usage_display = int(100*pct_usage)
	if small_prop is None:
		small_prop_display = small_prop
	else:
		small_prop_display = int(100*small_prop)

	if split_num is None:
		filename = '{}-{}-{}-{}-{}.pickle'.format(
			pct_usage_display, small_label, small_prop_display, seed, 0)
	else:
		filename = '{}-{}-{}-{}-{}.pickle'.format(
			pct_usage_display, small_label, small_prop_display, seed, split_num)		
	filepath = os.path.join(aug_data_path, filename)
	with open(filepath, 'rb') as f:
		data = pickle.load(f)
	# data = [(label, seq, aug) for label, seq, aug in data 
	# 		if label == small_label]
	if small_label is not None:
		assert all([label == small_label for label, _, _ in data])
	return data

def get_sst(input_length, aug_mode, pct_usage=None, 
			small_label=None, small_prop=None, seed=None, tokenizer=None,
			split_num=0):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	data_path = os.path.join(data_parent,'sst')
	data_dict = {}
	if aug_mode is None:
		for set_name in ['train', 'dev', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			data_dict[set_name] = read_no_aug(
				set_path, input_length, False, None)
		return data_dict
	elif aug_mode == 'synonym':
		aug_data_path = os.path.join(data_path, 'syn_aug')
		for set_name in ['train', 'dev', 'test']:
			set_path = os.path.join(aug_data_path, set_name+'.pickle')
			with open(set_path, 'rb') as f:
				data_dict[set_name] = pickle.load(f)
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
				set_path, input_length, False, None)
		if pct_usage is not None:
			train = read_context_aug(
				aug_data_path, pct_usage, small_label, small_prop, seed)
			print(Counter([tup[0] for tup in train]))
			data_dict['train'] = train
		else:
			train_small_label = read_context_aug(
				aug_data_path, pct_usage, small_label, small_prop, seed)
			set_path = os.path.join(data_path, 'train.txt')
			train_other_labels = read_no_aug(set_path, input_length, 
											False, small_label)
			train_other_labels = [(label, tokenizer.encode(
										seq, add_special_tokens=False), aug) 
								  for label, seq, aug in train_other_labels]
			data_dict['train'] = train_small_label + train_other_labels

		# for split, data in data_dict.items():
		# 	print(split, Counter([label for label, _, _ in data]))
		# exit()
			
		return data_dict
	else:
		raise ValueError('Unrecognized augmentation.')

def get_subj(input_length, aug_mode, pct_usage=None, small_label=None, 
			 small_prop=None, seed=0, tokenizer=None, gen_splits=True,
			 split_num=0):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	data_path = os.path.join(data_parent, 'subj')
	if aug_mode is None:
		file_path = os.path.join(data_path,'subj.txt')
		all_data = read_no_aug(file_path, input_length, True, None)
		# let 10% of data be the development set
		# dev_data, train_data = partition_within_classes(all_data, 0.1, False)
		# return {'dev': dev_data, 'train': train_data}
	elif aug_mode == 'synonym':
		aug_file_path = os.path.join(data_path, 'syn_aug/subj.pickle')
		with open(aug_file_path, 'rb') as f:
			all_data = pickle.load(f)
	elif aug_mode == 'trans':
		aug_file_path = os.path.join(data_path, 'trans_aug/subj.txt')
		all_data = read_trans_aug(aug_file_path)
	elif aug_mode == 'context':
		file_path = os.path.join(data_path,'subj.txt')
		all_data = read_no_aug(file_path, input_length, True, None)
	else:
		raise ValueError('Unrecognized augmentation.')
	if not gen_splits:
		return all_data
	dev_data, train_data = partition_within_classes(
		all_data, 0.1, False, seed=seed, split_num=split_num)
	if aug_mode == 'context':
		train_other_labels = [(label, tokenizer.encode(
									seq, add_special_tokens=False), aug) 
							  for label, seq, aug in train_data
							  if label != small_label]
		aug_data_path = os.path.join(data_path, 'context_aug/')
		train_small_label = read_context_aug(
			aug_data_path, pct_usage, small_label, small_prop, seed, 
			split_num=split_num)
		train_data = train_other_labels + train_small_label
		print(Counter([tup[0] for tup in train_data]))
		print(Counter([tup[0] for tup in dev_data]))
	# since subj uses crosstest we'll refer to dev set as test and dev
	return {'test': dev_data, 'dev': dev_data, 'train': train_data}

def get_sfu(input_length, aug_mode, pct_usage=None, small_label=None, 
			 small_prop=None, seed=0, tokenizer=None, gen_splits=True,
			 split_num=0):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	data_path = os.path.join(data_parent, 'sfu')
	if aug_mode is None:
		file_path = os.path.join(data_path,'sfu.txt')
		all_data = read_no_aug(file_path, input_length, False, None)
		# let 10% of data be the development set
		# dev_data, train_data = partition_within_classes(all_data, 0.1, False)
		# return {'dev': dev_data, 'train': train_data}
	elif aug_mode == 'synonym':
		aug_file_path = os.path.join(data_path, 'syn_aug/sfu.pickle')
		with open(aug_file_path, 'rb') as f:
			all_data = pickle.load(f)
	elif aug_mode == 'trans':
		aug_file_path = os.path.join(data_path, 'trans_aug/sfu.txt')
		all_data = read_trans_aug(aug_file_path)
	elif aug_mode == 'context':
		file_path = os.path.join(data_path,'sfu.txt')
		all_data = read_no_aug(file_path, input_length, False, None)
	else:
		raise ValueError('Unrecognized augmentation.')
	if not gen_splits:
		return all_data
	dev_data, train_data = partition_within_classes(
		all_data, 0.1, False, seed=seed, split_num=split_num)
	if aug_mode == 'context':
		train_other_labels = [(label, tokenizer.encode(
									seq, add_special_tokens=False), aug) 
							  for label, seq, aug in train_data
							  if label != small_label]
		aug_data_path = os.path.join(data_path, 'context_aug/')
		train_small_label = read_context_aug(
			aug_data_path, pct_usage, small_label, small_prop, seed, 
			split_num=split_num)
		train_data = train_other_labels + train_small_label
		print(Counter([tup[0] for tup in train_data]))
		print(Counter([tup[0] for tup in dev_data]))
	# since sfu uses crosstest we'll refer to dev set as test and dev
	return {'test': dev_data, 'dev': dev_data, 'train': train_data}