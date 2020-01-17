import logging
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

class BertAgent:
	def __init__(self, config, pct_usage=1, frac=0.5, geo=0.5):
		self.config = config
		self.pct_usage = pct_usage
		self.frac = frac
		self.geo = geo
		self.logger = logging.getLogger('BertAgent')
		self.cur_epoch = 0
		self.mngr = imdbDataManager(
			self.config, self.pct_usage, frac, geo)

		# self.mngr = ProConDataManager(
		# 	self.config, self.pct_usage, frac, geo)

		self.device = (torch.device('cuda:0' if torch.cuda.is_available() 
					   else 'cpu'))
		# print('Using '+str(int(100*self.pct_usage))+'% of the dataset.')
		# self.logger.info('Using '+str(self.pct_usage)+' of the dataset.')

		# if self.config.aug_mode == 'sr' or self.config.aug_mode == 'ca':
		# 	s = (str(int(100*self.frac))+'% of the training data will be ' 
		# 		 'original, the rest augmented.')
		# 	print_and_log(self.logger, s)
		# 	s = 'The geometric parameter is '+str(geo)+'.'
		# 	print_and_log(self.logger, s)

		self.checkpoint = 'bert-base-uncased'

	def initialize_model(self):
		self.model = BertTokenizer.from_pretrained(self.checkpoint)
		self.model = self.model.to(self.device)

		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in self.model.named_parameters() if
						not any(nd in n for nd in no_decay)], 
			 'weight_decay': self.weight_decay},
			{'params': [p for n, p in self.model.named_parameters() if 
						any(nd in n for nd in no_decay)], 
			 'weight_decay': 0.0}
		]
		# choosing AdamW defaults
		self.optimizer = AdamW(optimizer_grouped_parameters)
		# self.optimizer = AdamW()

		self.model.train()