import os
import random

import spacy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data import (get_split_indices, get_embeddings,
						get_sequences, sr_augment,
						ca_augment, partition_within_classes)
from dataset import BertDataset, RnnDataset

class TrecDatasetManager:
	def __init__(self, config, model, aug_mode, pct_usage, geo):
		self.config = config
		self.model = model
		self.aug_mode = aug_mode
		self.pct_usage = pct_usage
		self.geo = geo
		self.data_dict = get_trec()

		if model == 'rnn':
			nlp = spacy.load(
				'en_core_web_md', disable=['parser', 'tagger', 'ner'])
			nlp.vocab.set_vector(
				0, vector=np.zeros(nlp.vocab.vectors.shape[1]))
			self.nlp = nlp

	def get_train_set(self):
		if self.model == 'bert':
			return BertDataset(self.data_dict['train'], self.aug_mode, self.geo)
		elif self.model == 'rnn':
			return RnnDataset(self.data_dict['train'], self.nlp, self.aug_mode, self.geo)
		else:
			raise ValueError('Unrecognized model.')

def get_trec():
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	target_parent = os.path.join(repo_path, 'data')
	data_path = os.path.join(data_parent,'trec')
	data_dict = {}
	for set_name in ['train', 'test']:
		set_path = os.path.join(data_path, set_name+'.txt')
		set_data = []
		with open(set_path, 'rb') as f:
			for line in f.read().splitlines():
				line = line.decode('latin-1')
				label, example = line.split(' ', 1)
				label = int(label)
				set_data.append((label, example))
		data_dict[set_name] = set_data
	return data_dict