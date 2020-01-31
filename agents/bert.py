import numpy as np
import logging
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertTokenizer,
						  AdamW, get_linear_schedule_with_warmup)

from graphs.losses.loss import CrossEntropyLoss
from managers.sst import SSTDatasetManager
from managers.subj import SubjDatasetManager
from managers.trec import TrecDatasetManager
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper
from utils.logger import print_and_log

class BertAgent:
	def __init__(self, config, logger, data_name, input_length, max_epochs, 
				 aug_mode, mode, batch_size, small_label=None, 
				 small_prop=None, balance_seed=None, undersample=False, 
				 pct_usage=1, geo=0.5):
		assert not (undersample and aug_mode is not None), 'Cant undersample and augment'
		self.config = config
		self.logger = logger
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

		mngr_args = [self.config, 'bert', self.input_length, self.aug_mode,
				self.pct_usage, self.geo, self.batch_size]
		mngr_kwargs = {'small_label': self.small_label, 
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

		self.device = (torch.device('cuda:1' if torch.cuda.is_available() 
					   else 'cpu'))
		s = ('Model is Bert, dataset is {}, undersample is {}, aug mode is {}, geo is {},'
			' small_label is {} small_prop is {}, balance_seed is {}').format(
				data_name, self.undersample, self.aug_mode, self.geo, 
				self.small_label, self.small_prop, self.balance_seed)
		print_and_log(self.logger, s)

	def initialize_model(self):
		self.model = BertForSequenceClassification.from_pretrained(
			'bert-base-uncased', num_labels=self.num_labels).to(self.device)
		self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
		self.model.train()

	def run(self):
		if self.mode == 'crosstest':
			for fold in range(self.config.num_folds):
				self.initialize_model()
				s = 'Fold number {}'.format(fold)
				# print_and_log(self.logger, s)
				(self.train_loader, 
				self.val_loader) = self.mngr.crosstest_ldrs(fold)
				self.train()
				self.validate()
		elif self.mode == 'dev':
			self.train_loader, self.val_loader = self.mngr.get_dev_ldrs()
			self.initialize_model()
			total_steps = len(self.train_loader) * self.max_epochs
			self.scheduler = get_linear_schedule_with_warmup(
				self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
			self.train()
			# self.validate()
		else:
			raise ValueException('Unrecognized mode.')

	def train(self):
		if self.mode == 'crosstest':
			for self.cur_epoch in range(self.config.num_epochs):
				self.train_one_epoch()
			s = 'Stopped after ' + str(self.config.num_epochs) + ' epochs'
			# print_and_log(self.logger, s)

		elif self.mode == 'dev':
			# stopper = EarlyStopper(self.config.patience, 
			# 					   self.config.min_epochs)

			start_time = time.time()

			for self.cur_epoch in range(self.max_epochs):
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
		for x, y in tqdm(self.train_loader):
			attention_mask = (x > 0).float().to(self.device)
			x = x.to(self.device)
			y = y.to(self.device)
			self.model.zero_grad()
			current_loss, output = self.model(
				x, attention_mask=attention_mask, labels=y)
			loss.update(current_loss.item())
			current_loss.backward()
			MAX_GRAD_NORM = 1.0
			nn.utils.clip_grad_norm_(self.model.parameters(),
									 MAX_GRAD_NORM)
			self.optimizer.step()
			self.scheduler.step()
			# self.optimizer.zero_grad()
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
		loss = AverageMeter()
		acc = AverageMeter()
		for x, y in tqdm(self.val_loader):
			attention_mask = (x > 0).float().to(self.device)
			x = x.to(self.device)
			y = y.to(self.device)
			current_loss, output = self.model(
				x, attention_mask=attention_mask, labels=y)
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