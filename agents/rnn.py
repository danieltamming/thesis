import os
import logging
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F

from graphs.rnn import Rnn
from graphs.loss import CrossEntropyLoss
from data.managers import (SSTDatasetManager, SubjDatasetManager, 
						   TrecDatasetManager)
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper
from utils.logger import print_and_log

class RnnAgent:
	def __init__(self, device, logger, data_name, input_length, max_epochs, 
				 aug_mode, mode, batch_size, small_label=None, 
				 small_prop=None, balance_seed=None, undersample=False,
				 pct_usage=None, geo=0.5, verbose=False):
		assert not (undersample and aug_mode is not None), \
			   'Cant undersample and augment'
		assert sum([mode == 'save', 
					pct_usage is not None, 
					small_label is not None]) == 1, \
			   'Either saving, balancing, or trying on specific percentage'
		self.logger = logger
		self.data_name = data_name
		self.input_length = input_length
		self.max_epochs = max_epochs
		self.aug_mode = aug_mode
		self.mode = mode
		self.batch_size = batch_size
		self.small_label = small_label
		self.small_prop = small_prop
		self.balance_seed = balance_seed
		self.undersample = undersample
		self.pct_usage = pct_usage
		self.geo = geo
		self.verbose = verbose

		self.loss = CrossEntropyLoss()

		nlp = spacy.load(
			'en_core_web_md', disable=['parser', 'tagger', 'ner'])
		nlp.vocab.set_vector(
			0, vector=np.zeros(nlp.vocab.vectors.shape[1]))
		self.nlp = nlp

		mngr_args = ['rnn', self.input_length, self.aug_mode,
					 self.pct_usage, self.geo, self.batch_size]
		mngr_kwargs = {'nlp': self.nlp, 'small_label': self.small_label, 
					   'small_prop': self.small_prop, 
					   'balance_seed': self.balance_seed, 
					   'undersample': undersample}
		if data_name == 'sst':
			self.num_labels = 2
			self.mngr = SSTDatasetManager(*mngr_args, **mngr_kwargs)
		elif data_name == 'subj':
			self.num_labels = 2
			self.mngr = SubjDatasetManager(*mngr_args, **mngr_kwargs)
		elif data_name == 'trec':
			self.num_labels = 6
			self.mngr = TrecDatasetManager(*mngr_args, **mngr_kwargs)
		else:
			raise ValueError('Data name not recognized.')

		self.device = (torch.device(device if torch.cuda.is_available() 
					   else 'cpu'))

		s = ('Model is RNN, dataset is {}, undersample is {}, aug mode is {}, geo is {},'
			' pct_usage is {}, small_label is {}, small_prop is {}, balance_seed is {}').format(
				data_name, self.undersample, self.aug_mode, self.geo, self.pct_usage,
				self.small_label, self.small_prop, self.balance_seed)
		print_and_log(self.logger, s)

	def initialize_model(self):
		embed_arr = torch.from_numpy(self.nlp.vocab.vectors.data)
		self.model = Rnn(
			embed_arr, self.num_labels).to(self.device)
		self.optimizer = Adam(self.model.parameters())
		self.model.train()

	def save_checkpoint(self):
		ckpt_file = 'checkpoints/rnn-{}.pth'.format(self.data_name)
		torch.save(self.model.state_dict(), ckpt_file)

	def run(self):
		if self.mode == 'crosstest':
			raise NotImplementedError('Crosstest not implemented.')
		elif self.mode == 'dev':
			self.train_loader, self.val_loader = self.mngr.get_dev_ldrs()
			self.initialize_model()
			self.train()
			# self.validate()
		elif self.mode == 'save':
			self.train_loader, self.val_loader = self.mngr.get_dev_ldrs()
			self.initialize_model()
			self.train()
			self.save_checkpoint()
		else:
			raise ValueException('Unrecognized mode.')

	def train(self):
		start_time = time.time()
		if self.verbose:
			iterator = range(self.max_epochs)
		else:
			iterator = tqdm(range(self.max_epochs))
		for self.cur_epoch in iterator:
			self.train_one_epoch()
			acc,_ = self.validate()

			if start_time is not None:
				print('{} s/it'.format(round(time.time()-start_time,3)))
				start_time = None

			# if stopper.update_and_check(acc, printing=True): 
			# 	s = ('Stopped early with patience '
			# 		'{}'.format(self.config.patience))
			# 	print_and_log(self.logger, s)
			# 	break

	def train_one_epoch(self):
		self.model.train()
		loss = AverageMeter()
		acc = AverageMeter()
		if self.verbose:
			iterator = tqdm(self.train_loader)
		else:
			iterator = self.train_loader
		for x, y in iterator:
			x = x.to(self.device)
			y = y.to(self.device)
			output = self.model(x)
			current_loss = self.loss(output, y)
			self.optimizer.zero_grad()
			current_loss.backward()
			self.optimizer.step()
			loss.update(current_loss.item())
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])
		# if self.mode == 'crossval':
		s = ('Training epoch {} | loss: {} - accuracy: ' 
		'{}'.format(self.cur_epoch, 
					round(loss.val, 5), 
					round(acc.val, 5)))
		print_and_log(self.logger, s)
		# self.logger.info(s)
		# print(s)

	def validate(self):
		self.model.eval()
		with torch.no_grad():
			loss = AverageMeter()
			acc = AverageMeter()
			# for x, y in tqdm(self.val_loader):
			for x, y in self.val_loader:
				x = x.to(self.device)
				y = y.to(self.device)
				output = self.model(x)
				current_loss = self.loss(output, y)
				loss.update(current_loss.item())
				accuracy = get_accuracy(output, y)
				acc.update(accuracy, y.shape[0])
			s = ('Validating epoch {} | loss: {} - accuracy: ' 
				'{}'.format(self.cur_epoch, 
							round(loss.val, 5), 
							round(acc.val, 5)))
		print_and_log(self.logger, s)
		# self.logger.info(s)
		# print(s)

		return acc.val, loss.val