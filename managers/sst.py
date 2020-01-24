import os
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset import BertDataset, RnnDataset

class SSTDatasetManager:
	def __init__(self, config, model_type, input_length, 
				 aug_mode, pct_usage, geo, batch_size, nlp=None,
				 small_label=None, small_prop=None):
		self.config = config
		self.model_type = model_type
		self.input_length = input_length
		self.aug_mode = aug_mode
		self.pct_usage = pct_usage
		self.geo = geo
		self.batch_size = batch_size
		self.small_label = small_label
		self.small_prop = small_prop
		self.data_dict = get_sst(self.input_length)
		
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
			small_label = self.small_label
			small_prop = self.small_prop
		else:
			# val and test splits have no augmentation
			aug_mode = None
			geo = None
			small_label = None
			small_prop = None
		if self.model_type == 'bert':
			return BertDataset(self.data_dict[split_key], self.input_length, 
							   aug_mode, geo)
		elif self.model_type == 'rnn':
			return RnnDataset(self.data_dict[split_key], self.input_length, 
							  self.nlp, aug_mode, geo, small_label, small_prop)
		else:
			raise ValueError('Unrecognized model type.')

def get_sst(input_length):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	target_parent = os.path.join(repo_path, 'data')
	data_path = os.path.join(data_parent,'sst')
	data_dict = {}
	for set_name in ['train', 'dev', 'test']:
		set_path = os.path.join(data_path, set_name+'.txt')
		set_data = []
		with open(set_path) as f:
			for line in f.read().splitlines():
				label, example = line.split(maxsplit=1)
				label = int(label)
				example = ' '.join(example.split()[:input_length])
				set_data.append((label, example))
		data_dict[set_name] = set_data
	return data_dict

# CONFIRM EXAMPLE ISN'T BEING WEIRD NOW. BEFORE THE INPUT LENGTH CUTOFF HERE
# THE MODEL WAS ACHIEVING BETTER ACCURACY, ALTHOUGH THAT COULD JUST BE BY CHANCE