import numpy as np
import logging
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from graphs.losses.loss import CrossEntropyLoss
from managers.sst import SSTDatasetManager
from managers.subj import SubjDatasetManager
from managers.trec import TrecDatasetManager
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper
from utils.logger import print_and_log

class BertAgent:
	def __init__(self, config, data_name, input_length, 
				 aug_mode, mode, batch_size, pct_usage=1, geo=0.5):
		self.config = config
		self.input_length = input_length
		self.aug_mode = aug_mode
		self.mode = mode
		self.batch_size = batch_size
		self.pct_usage = pct_usage
		self.geo = geo
		# self.logger = logging.getLogger('BertAgent')
		self.cur_epoch = 0
		self.loss = CrossEntropyLoss()

		self.MAX_EPOCHS = 500

		if data_name == 'sst':
			self.mngr = SSTDatasetManager(
				self.config, 'bert', self.input_length, self.aug_mode,
				self.pct_usage, self.geo, self.batch_size)
		elif data_name == 'subj':
			self.mngr = SubjDatasetManager(
				self.config, 'bert', self.input_length, self.aug_mode,
				self.pct_usage, self.geo, self.batch_size)
		elif data_name == 'trec':
			self.mngr = TrecDatasetManager(
				self.config, 'bert', self.input_length, self.aug_mode,
				self.pct_usage, self.geo, self.batch_size)
		else:
			raise ValueError('Data name not recognized.')
		self.device = (torch.device('cuda:0' if torch.cuda.is_available() 
					   else 'cpu'))
		# print('Using '+str(int(100*self.pct_usage))+'% of the dataset.')
		# self.logger.info('Using '+str(self.pct_usage)+' of the dataset.')

		# if self.config.aug_mode == 'sr' or self.config.aug_mode == 'ca':
		# 	s = 'The geometric parameter is '+str(geo)+'.'
		# 	print_and_log(self.logger, s)

	def initialize_model(self):
		self.model = BertForSequenceClassification.from_pretrained(
			'bert-base-uncased')
		self.model = self.model.to(self.device)
		# ------------------------------------------------
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in self.model.named_parameters() if
						not any(nd in n for nd in no_decay)], 
			 'weight_decay': 0.0},
			{'params': [p for n, p in self.model.named_parameters() if 
						any(nd in n for nd in no_decay)], 
			 'weight_decay': 0.0}
		]
		self.optimizer = AdamW(optimizer_grouped_parameters)
		# --------------------------------------------
		# self.optimizer = AdamW(self.model.named_parameters())
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

			for self.cur_epoch in range(self.MAX_EPOCHS):
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
		# -----------------------------------
		# TEST THAT MDOEL IS ACTUALLY LEARNING
		# -----------------------------------
		self.model.train()
		loss = AverageMeter()
		acc = AverageMeter()
		for x, y in tqdm(self.train_loader):
			attention_mask = (x > 0).float().to(self.device)
			x = x.to(self.device)
			y = y.to(self.device)
			self.optimizer.zero_grad()
			current_loss, output = self.model(
				x, attention_mask=attention_mask, labels=y)
			current_loss.backward()
			loss.update(current_loss.item())
			MAX_GRAD_NORM = 1.0
			nn.utils.clip_grad_norm_(self.model.parameters(),
									 MAX_GRAD_NORM)
			self.optimizer.step()
			# self.optimizer.zero_grad()
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])
		# if self.mode == 'crossval':
		s = ('Training epoch {} | loss: {} - accuracy: ' 
		'{}'.format(self.cur_epoch, 
					round(loss.val, 5), 
					round(acc.val, 5)))
		# print_and_log(self.logger, s)
		# self.logger.info(s)
		print(s)

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
		# print_and_log(self.logger, s)
		# self.logger.info(s)
		print(s)

		return acc.val, loss.val