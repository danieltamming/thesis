import os
import numpy as np
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import (BertForSequenceClassification,
						  AdamW, get_linear_schedule_with_warmup, 
						  get_constant_schedule)

from graphs.loss import CrossEntropyLoss
from data.datasets import TestTimeDataset
from data.managers import (SSTDatasetManager, SubjDatasetManager, 
						   SFUDatasetManager)
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper
from utils.logger import print_and_log

class BertAgent:
	def __init__(self, device, logger, data_name, input_length, max_epochs, 
				 lr, aug_mode, mode, batch_size, accumulation_steps, 
				 small_label=None, small_prop=None, balance_seed=None, 
				 undersample=False, pct_usage=None, geo=0.5, split_num=0,
				 verbose=False):
		assert not (undersample and aug_mode is not None), \
			   'Cant undersample and augment'
		assert sum([mode == 'test-aug', mode == 'save', pct_usage is not None, 
					small_label is not None]) == 1, \
			   'Either saving, balancing, or trying on specific percentage' 
		self.logger = logger
		self.data_name = data_name
		self.input_length = input_length
		self.max_epochs = max_epochs
		self.lr = lr
		self.aug_mode = aug_mode
		self.mode = mode
		self.batch_size = batch_size
		self.accumulation_steps = accumulation_steps
		self.small_label = small_label
		self.small_prop = small_prop
		self.balance_seed = balance_seed
		self.undersample = undersample
		self.pct_usage = pct_usage
		self.geo = geo
		self.split_num = split_num
		self.verbose = verbose

		mngr_args = ['bert', self.input_length, self.aug_mode,
				self.pct_usage, self.geo, self.batch_size]
		mngr_kwargs = {'small_label': self.small_label, 
				  'small_prop': self.small_prop,
				  'balance_seed': self.balance_seed, 
				  'undersample': undersample,
				  'split_num': self.split_num}
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
		s = ('Model is Bert, dataset is {}, undersample is {},'
			 ' aug mode is {}, geo is {}, pct_usage is {}, small_label is {},'
			 ' small_prop is {}, balance_seed is {}, lr is {},'
			 ' max_epochs is {}, split_num is {}').format(
				data_name, self.undersample, self.aug_mode, self.geo, 
				self.pct_usage, self.small_label, self.small_prop, 
				self.balance_seed, self.lr, self.max_epochs, self.split_num)
		print_and_log(self.logger, s)

	def initialize_model(self):
		self.model = BertForSequenceClassification.from_pretrained(
			'bert-base-uncased', num_labels=self.num_labels).to(self.device)
		# optimizer immediately below had worse performance
		# self.optimizer = AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)
		self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
		total_steps = (len(self.train_loader) 
					   // self.accumulation_steps) * self.max_epochs
		# self.scheduler = get_constant_schedule(self.optimizer)
		# scheduler immediately below had worse performance
		# self.scheduler = get_linear_schedule_with_warmup(
		# 	self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
		self.scheduler = get_linear_schedule_with_warmup(
			self.optimizer, num_warmup_steps=0.1*total_steps, 
			num_training_steps=total_steps)
		self.model.train()

	def save_checkpoint(self):
		ckpt_dir = 'checkpoints/bert-' + self.data_name
		if not os.path.exists(ckpt_dir):
			os.mkdir(ckpt_dir)
		self.model.save_pretrained(ckpt_dir)

	def run(self):
		if self.mode == 'crosstest':
			raise NotImplementedError('Crosstest not implemented.')
		elif self.mode in ['dev', 'test']:
			self.train_loader, self.val_loader = self.mngr.get_dev_ldrs('dev')
			self.initialize_model()
			self.train()
			if self.mode == 'test':
				acc, _ = self.validate()
			# self.validate()
		# elif self.mode == 'save':
		# 	self.train_loader, self.val_loader = self.mngr.get_dev_ldrs()
		# 	self.initialize_model()
		# 	self.train()
		# 	self.save_checkpoint()
		elif self.mode == 'test-aug':
			ckpt_dir = 'checkpoints/bert-' + self.data_name
			self.model = BertForSequenceClassification.from_pretrained(
				ckpt_dir, num_labels=self.num_labels).to(self.device)
			self.model.eval()
			self.test_aug()
		else:
			raise ValueException('Unrecognized mode.')

	def test_aug(self):
		orig_correct = []
		aug_correct = []
		# implement test time augmentation
		dataset = TestTimeDataset(
			self.data_name, self.input_length, self.aug_mode)
		for seqs, weights, label in tqdm(dataset):
			attention_mask = (seqs > 0).float().to(self.device)
			seqs = seqs.to(self.device)
			output = self.model(seqs, attention_mask)[0].data
			orig_pred = torch.max(output[0,:], 0).indices.item()
			probs = F.softmax(output, dim=1)
			weighted_prob = torch.mv(probs.T, weights.float())
			aug_pred = torch.max(weighted_prob, 0).indices.item()
			
			orig_correct.append(int(label == orig_pred))
			aug_correct.append(int(label == aug_pred))

		print(orig_correct)
		print(aug_correct)

	def train(self):
		# stopper = EarlyStopper(self.config.patience, 
		# 					   self.config.min_epochs)

		start_time = time.time()
		if self.verbose:
			iterator = range(self.max_epochs)
		else:
			iterator = tqdm(range(self.max_epochs))
		for self.cur_epoch in iterator:
			self.train_one_epoch()
			if self.mode != 'test':
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
		self.optimizer.zero_grad()
		self.model.train()
		loss = AverageMeter()
		acc = AverageMeter()
		if self.verbose:
			iterator = enumerate(tqdm(self.train_loader))
		else:
			iterator = enumerate(self.train_loader)
		for i, (x, y) in iterator:
			attention_mask = (x > 0).float().to(self.device)
			x = x.to(self.device)
			y = y.to(self.device)
			current_loss, output = self.model(
				x, attention_mask=attention_mask, labels=y)
			current_loss = current_loss / self.accumulation_steps
			current_loss.backward()
			loss.update(current_loss.detach().item())
			# MAX_GRAD_NORM = 1.0
			# nn.utils.clip_grad_norm_(self.model.parameters(),
			# 						 MAX_GRAD_NORM)
			if (i+1) % self.accumulation_steps == 0:
				self.optimizer.step()
				self.scheduler.step()
				self.optimizer.zero_grad()

			output = output.detach().cpu().numpy()
			y = y.cpu().numpy()
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])

			# del current_loss
			# del output
			# del accuracy
			# del attention_mask
		# if self.mode == 'crossval':
		s = ('Training epoch {} | loss: {} - accuracy: ' 
		'{}'.format(self.cur_epoch, 
					round(loss.val, 5), 
					round(acc.val, 5)))
		print_and_log(self.logger, s)

		# del loss
		# del acc
		# self.logger.info(s)
		# print(s)

	def validate(self):
		self.model.eval()
		with torch.no_grad():
			loss = AverageMeter()
			acc = AverageMeter()
			for x, y in self.val_loader:
				attention_mask = (x > 0).float().to(self.device)
				x = x.to(self.device)
				y = y.to(self.device)
				current_loss, output = self.model(
					x, attention_mask=attention_mask, labels=y)
				loss.update(current_loss.detach().item())
				output = output.detach().cpu().numpy()
				y = y.cpu().numpy()
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