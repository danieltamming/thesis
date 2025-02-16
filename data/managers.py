import os
import random
from collections import Counter

import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils.data import get_sst, get_subj, get_sfu, partition_within_classes
from data.datasets import BertDataset, RnnDataset

class DatasetManagerBase:
	def __init__(self, data_func, model_type, input_length, 
				 aug_mode, pct_usage, geo, batch_size, nlp=None,
				 small_label=None, small_prop=None, balance_seed=None,
				 undersample=False, split_num=0):
		self.model_type = model_type
		self.input_length = input_length
		self.aug_mode = aug_mode
		self.pct_usage = pct_usage
		self.geo = geo
		self.batch_size = batch_size
		self.small_label = small_label
		self.small_prop = small_prop
		self.balance_seed = balance_seed
		self.undersample = undersample
		self.split_num = split_num
		if model_type == 'bert' or aug_mode == 'context':
			self.tokenizer = BertTokenizer.from_pretrained(
				'bert-base-uncased')
		else:
			self.tokenizer = None
		if aug_mode == 'context':
			kwargs = {'pct_usage': pct_usage, 'small_label': small_label,
					  'small_prop': small_prop, 'seed': balance_seed,
					  'tokenizer': self.tokenizer, 'split_num':split_num}
		else:
			kwargs = {'seed': balance_seed, 'tokenizer': self.tokenizer,
					  'split_num': split_num}
		self.data_dict = data_func(self.input_length, self.aug_mode, **kwargs)

		# for split, data in self.data_dict.items():
		# 	print(split, Counter([label for label, _, _ in data]))
		# exit()

		if model_type == 'rnn':
			assert nlp is not None
			self.nlp = nlp

	def get_dev_ldrs(self, val_type):
		# val type is 'dev' for development set or 'test' for test set
		train_dataset = self.get_dataset('train')
		train_loader = DataLoader(
			train_dataset, self.batch_size, pin_memory=True, shuffle=True)
		val_dataset = self.get_dataset(val_type)

		print('train', Counter([label for label, _, _ in train_dataset.data]))
		print('valid', Counter([label for label, _, _ in val_dataset.data]))
		# import pickle
		# with open('test_split_{}.pickle'.format(self.split_num), 'wb') as f:
		# 	pickle.dump(val_dataset.data, f, protocol=pickle.HIGHEST_PROTOCOL)		
		# exit()
		
		val_loader = DataLoader(
			val_dataset, self.batch_size, pin_memory=True)
		return train_loader, val_loader

	def get_dataset(self, split_key):
		if split_key == 'train':
			aug_mode = self.aug_mode
			geo = self.geo
			small_label = self.small_label
			small_prop = self.small_prop
			pct_usage = self.pct_usage
		else:
			# val and test splits have no augmentation
			aug_mode = None
			geo = None
			small_label = None
			small_prop = None
			pct_usage = None
		args = [self.data_dict[split_key], self.input_length, aug_mode]
		kwargs = {'pct_usage': pct_usage, 'geo': geo, 'small_label': small_label, 
				  'small_prop': small_prop, 'balance_seed': self.balance_seed,
				  'undersample': self.undersample, 'tokenizer': self.tokenizer}
		if self.model_type == 'bert':
			return BertDataset(*args, **kwargs)
		elif self.model_type == 'rnn':
			return RnnDataset(self.nlp, *args, **kwargs)
		else:
			raise ValueError('Unrecognized model type.')

	def get_train_inf_data(self):
		'''
		Used for BERT contextual augmentation file creation
		'''
		args = [self.data_dict['train'], self.input_length, None]
		kwargs = {'pct_usage': self.pct_usage, 
				  'geo': self.geo, 
				  'small_label': self.small_label, 
				  'small_prop': self.small_prop, 'balance_seed': self.balance_seed,
				  'undersample': self.undersample, 'tokenizer': self.tokenizer}		
		dataset = BertDataset(
			self.data_dict['train'], self.input_length, None,
			pct_usage=self.pct_usage, small_label=self.small_label,
			small_prop=self.small_prop, balance_seed=self.balance_seed, 
			keep_inf_data=True)
		train_data = [(label, seq) for label, seq, _ in dataset.data]
		inf_data = [(label, seq) for label, seq, _ in dataset.inf_data]
		return train_data, inf_data

class SSTDatasetManager(DatasetManagerBase):
	def __init__(self, *args, **kwargs):
		super().__init__(get_sst, *args, **kwargs)

class SubjDatasetManager(DatasetManagerBase):
	def __init__(self, *args, **kwargs):
		super().__init__(get_subj, *args, **kwargs)

class SFUDatasetManager(DatasetManagerBase):
	def __init__(self, *args, **kwargs):
		super().__init__(get_sfu, *args, **kwargs)
