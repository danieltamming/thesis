import os
import random

import numpy as np
from torch.utils.data import DataLoader

from utils.data import partition_within_classes
from dataset import BertDataset, RnnDataset

class SubjDatasetManager:
	def __init__(self, config, model_type, input_length, 
				 aug_mode, pct_usage, geo, batch_size, nlp=None):
		self.config = config
		self.model_type = model_type
		self.input_length = input_length
		self.aug_mode = aug_mode
		self.pct_usage = pct_usage
		self.geo = geo
		self.batch_size = batch_size
		self.data_dict = get_subj(input_length)

		if model_type == 'rnn':
			assert nlp is not None
			self.nlp = nlp

	def get_dev_ldrs(self):
		train_dataset = self.get_dataset('train')
		train_loader = DataLoader(
			train_dataset, self.batch_size, pin_memory=True, shuffle=True)
		val_dataset = self.get_dataset('dev')
		val_loader = DataLoader(
			val_dataset, self.batch_size, pin_memory=True)
		return train_loader, val_loader

	def get_dataset(self, split_key):
		if split_key == 'train':
			aug_mode = self.aug_mode
			geo = self.geo
		else:
			# val and test splits have no augmentation
			aug_mode = None
			geo = None
		if self.model_type == 'bert':
			return BertDataset(self.data_dict[split_key], 
							   self.input_length, aug_mode, geo)
		elif self.model_type == 'rnn':
			return RnnDataset(self.data_dict[split_key], self.input_length, 
							  self.nlp, aug_mode, geo)
		else:
			raise ValueError('Unrecognized model type.')

def get_subj(input_length):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	target_parent = os.path.join(repo_path, 'data')
	file_path = os.path.join(data_parent,'subj/subj.txt')
	all_data = []
	with open(file_path, 'rb') as f:
		for line in f.read().splitlines():
			line = line.decode('latin-1')
			label, example = line.split(maxsplit=1)
			label = int(label)
			example = ' '.join(example.split()[:input_length])
			all_data.append((label, example))
	# let 10% of data be the development set
	dev_data, train_data = partition_within_classes(all_data, 0.1)
	return {'dev': dev_data, 'train': train_data}