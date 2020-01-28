import numpy as np
import logging
import time

import spacy
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from graphs.models.bilstm import BiLSTM
from graphs.models.rnn import Rnn
from graphs.losses.loss import CrossEntropyLoss
from managers.sst import SSTDatasetManager
from managers.subj import SubjDatasetManager
from managers.trec import TrecDatasetManager
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper
from utils.logger import print_and_log

class RnnAgent:
	def __init__(self, config, data_name, input_length, max_epochs, 
				 aug_mode, mode, batch_size, small_label=None, 
				 small_prop=None, balance_seed=None, undersample=False,
				 pct_usage=1, geo=0.5):
		self.config = config
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

		self.logger = logging.getLogger('RnnAgent')
		self.loss = CrossEntropyLoss()

		nlp = spacy.load(
			'en_core_web_md', disable=['parser', 'tagger', 'ner'])
		nlp.vocab.set_vector(
			0, vector=np.zeros(nlp.vocab.vectors.shape[1]))
		self.nlp = nlp

		args = [self.config, 'rnn', self.input_length, self.aug_mode,
				self.pct_usage, self.geo, self.batch_size]
		kwargs = {'nlp': self.nlp, 'small_label': self.small_label, 
				  'small_prop': self.small_prop, 
				  'balance_seed': self.balance_seed, 
				  'undersample': undersample}
		if data_name == 'sst':
			self.num_labels = 2
			self.mngr = SSTDatasetManager(*args, **kwargs)
		elif data_name == 'subj':
			self.num_labels = 2
			self.mngr = SubjDatasetManager(*args, **kwargs)
		elif data_name == 'trec':
			self.num_labels = 6
			self.mngr = TrecDatasetManager(*args, **kwargs)
		else:
			raise ValueError('Data name not recognized.')

		self.device = (torch.device('cuda:1' if torch.cuda.is_available() 
					   else 'cpu'))
		# print('Using '+str(int(100*self.pct_usage))+'% of the dataset.')
		# self.logger.info('Using '+str(self.pct_usage)+' of the dataset.')

		# if self.config.aug_mode == 'sr' or self.config.aug_mode == 'ca':
		# 	s = 'The geometric parameter is '+str(geo)+'.'
		# 	print_and_log(self.logger, s)
		s = 'Aug mode is {}, geo is {}, small_label is {} small_prop is {}'.format(self.aug_mode, self.geo, self.small_label, self.small_prop)
		print_and_log(self.logger, s)

	def initialize_model(self):
		embed_arr = torch.from_numpy(self.nlp.vocab.vectors.data)
		self.model = Rnn(
			self.config, embed_arr, self.num_labels).to(self.device)
		self.optimizer = Adam(self.model.parameters())
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
		loss = AverageMeter()
		acc = AverageMeter()
		for x, y in tqdm(self.val_loader):
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