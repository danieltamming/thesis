import os
import sys
import inspect
import random
import pickle
import multiprocessing as mp
from collections import Counter
from itertools import cycle, islice

current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (BertForMaskedLM, BertTokenizer, 
						  AdamW, get_linear_schedule_with_warmup)

from utils.data import get_sst, get_subj, get_trec, partition_within_classes
from utils.metrics import AverageMeter
from utils.parsing import get_device
from data.managers import SSTDatasetManager, SubjDatasetManager


def mask_tokens(inputs, tokenizer):
	mlm_prob = 0.15
	targets = inputs.clone()
	prob_matrix = torch.full(inputs.shape, mlm_prob)
	special_tokens_mask = np.isin(inputs.numpy(), tokenizer.all_special_ids)
	prob_matrix.masked_fill_(torch.from_numpy(special_tokens_mask), value=0)
	masked_indices = torch.bernoulli(prob_matrix).bool()
	targets[~masked_indices] = -1
	# -------------------------------------------
	# new add for bug fix. check docs to see whether masked should be -1 or 0
	targets[~masked_indices] = 0
	# ----------------------------------------
	indices_replaced = (
		torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() 
		& masked_indices)
	inputs[indices_replaced] = tokenizer.mask_token_id
	indices_random = (
		torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() 
		& masked_indices 
		& ~indices_replaced)
	random_words = torch.randint(
		len(tokenizer), inputs.shape, dtype=torch.long)
	inputs[indices_random] = random_words[indices_random]
	return inputs, targets

def get_attention_mask(inputs):
	return (inputs > 0).float()

def cat_to_token_type(categories, seq_len):
	'''
	Returns batch_size x seq_len Tensor with values at [i,:] all
	equal to the category id. This provides the change from segment
	embeddings to category embeddings - thus chaning from contextual
	augmentation to conditional contextual augmentation
	'''
	return torch.ger(categories, torch.ones(seq_len, dtype=torch.long))

def mask_gen(seq, tokenizer):
	'''
	A generator that returns the masked index 
	and sequence with that index masked.
	It does not return partial words
	such as '##asse', nor does it return special tokens
	'''
	for i in range(seq.shape[0]):
		num = seq[i].item()
		s = tokenizer.convert_ids_to_tokens(num)
		if num in tokenizer.all_special_ids or not s.isalpha():
			continue
		seq[i] = tokenizer.mask_token_id
		yield i, seq
		seq[i] = num


class TextDataset(Dataset):
	def __init__(self, data, tokenizer, input_length):
		self.examples = []
		for label, text in data:
			tokenized_text = tokenizer.tokenize(text)
			ids_text = tokenizer.convert_tokens_to_ids(tokenized_text)
			input_ids = tokenizer.build_inputs_with_special_tokens(
				ids_text)
			input_ids = input_ids[:input_length]
			input_ids.extend((input_length - len(input_ids))
							 * [tokenizer.pad_token_id])
			self.examples.append((input_ids, label))
		# self.examples = make_length(self.examples, tokenizer, input_length)
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		'''
		May need to convert one or both elements of this tuple to torch tensor
		'''
		sequence, label = self.examples[idx]
		return torch.tensor(sequence), label


class BertAgent:
	def __init__(self, lr, data_name, seed, pct_usage, 
				 small_label, small_prop, split_num=0):
		self.data_name = data_name
		self.seed = seed
		self.pct_usage = pct_usage
		self.small_label = small_label
		self.small_prop = small_prop
		self.split_num = split_num
		self.lr = lr
		self.input_length = 80
		self.batch_size = 16
		self.num_train_epochs = 9
		self.checkpoint = 'bert-base-uncased'
		self.output_dir = 'bert-ckpt'
		self.weight_decay = 0.0
		self.max_grad_norm = 1.0
		self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
		self.model = BertForMaskedLM.from_pretrained(self.checkpoint)
		self.device = (torch.device(get_device() if torch.cuda.is_available() 
					   else 'cpu'))
		self.model = self.model.to(self.device)
		mngr_args = ['bert', self.input_length, None,
					 self.pct_usage, -1, self.batch_size]
		mngr_kwargs = {'small_label': self.small_label, 
					   'small_prop': self.small_prop, 
					   'balance_seed': self.seed,
					   'split_num': self.split_num}
		if data_name == 'sst':
			self.num_labels = 2
			self.mngr = SSTDatasetManager(*mngr_args, **mngr_kwargs)
		elif data_name == 'subj':
			self.num_labels = 2
			self.mngr = SubjDatasetManager(*mngr_args, **mngr_kwargs)
		else:
			raise ValueError('Data name not recognized.')
		if self.small_label is not None  and self.small_prop is not None:
			self.train_data, self.inf_data = self.mngr.get_train_inf_data()
		elif self.pct_usage is not None:
			self.train_data = [(label, seq) for label, seq, aug 
								in self.mngr.get_dataset('train').data]
			self.inf_data = self.train_data
		else:
			ValueError('Invalid input.')

		# with open('split_{}.pickle'.format(self.split_num), 'wb') as f:
		# 	pickle.dump(self.train_data, f, protocol=pickle.HIGHEST_PROTOCOL)		

	def save_checkpoint(self):
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		self.model.save_pretrained(self.output_dir)
		self.tokenizer.save_pretrained(self.output_dir)

	def train(self):
		self.dataset = TextDataset(self.train_data, self.tokenizer, 
								   self.input_length)
		self.loader = DataLoader(self.dataset, self.batch_size, 
								 pin_memory=True, shuffle=True)
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
		self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
		total_steps = len(self.loader) * self.num_train_epochs
		self.scheduler = get_linear_schedule_with_warmup(
			self.optimizer, num_warmup_steps=0.1*total_steps, 
			num_training_steps=total_steps
			)
		self.model.train()
		self.model.zero_grad()
		for epoch in range(self.num_train_epochs):
			epoch_loss = self.train_one_epoch()
			epoch_loss = round(epoch_loss, 5)
			print('Epoch {} | loss = {}'.format(epoch, epoch_loss))

	def train_one_epoch(self):
		loss_meter = AverageMeter()
		for batch, categories in tqdm(self.loader):
			token_type_ids = cat_to_token_type(categories, batch.shape[1])
			attention_mask = get_attention_mask(batch)
			inputs, targets = mask_tokens(batch, self.tokenizer)
			inputs = inputs.to(self.device)
			attention_mask = attention_mask.to(self.device)
			token_type_ids = token_type_ids.to(self.device)
			targets = targets.to(self.device)
			loss, prediction_scores = self.model(
				inputs, attention_mask=attention_mask, 
				token_type_ids=token_type_ids, masked_lm_labels=targets)

			# --------------------------------------------------------
			_, pred = torch.max(prediction_scores, 2)
			pred = torch.mul(pred, attention_mask.long())
			# --------------------------------------------------------
			loss_meter.update(loss.item())
			loss.backward()
			# nn.utils.clip_grad_norm_(self.model.parameters(), 
			# 						 self.max_grad_norm)
			self.optimizer.step()
			self.scheduler.step()
			self.model.zero_grad()
		return loss_meter.val

	def augment(self):
		self.dataset = TextDataset(self.inf_data, self.tokenizer, 
								   self.input_length)
		self.loader = DataLoader(self.dataset, self.batch_size, 
								 pin_memory=True, shuffle=False)
		self.model.eval()
		data = []
		for seq, cat in tqdm(self.dataset):
			aug = {}
			for i, masked_seq in mask_gen(seq, self.tokenizer):
				batch = torch.unsqueeze(masked_seq, 0)
				token_type_ids = cat*torch.ones(batch.shape[1], 
												dtype=torch.long)
				# token_type_ids = cat_to_token_type(cat, batch.shape[1])
				attention_mask = get_attention_mask(batch)
				batch = batch.to(self.device)
				attention_mask = attention_mask.to(self.device)
				token_type_ids = token_type_ids.to(self.device)
				prediction_scores = self.model(
					batch, attention_mask=attention_mask, 
					token_type_ids=token_type_ids)[0].data
				ith_preds = prediction_scores[0,i,:]
				top_10_ids = torch.topk(ith_preds, 10)[1].tolist()
				top_10_toks = self.tokenizer.convert_ids_to_tokens(top_10_ids)
				top_toks = [s for s in top_10_toks if s.isalpha()]
				top_ids = self.tokenizer.convert_tokens_to_ids(top_toks)
				if len(top_ids) > 0:
					aug[i-1] = top_ids
			clean_seq = [eyedee for eyedee in seq.tolist() if eyedee 
						 not in self.tokenizer.all_special_ids]
			assert clean_seq == seq.tolist()[1:1+len(clean_seq)]
			data.append((cat, clean_seq, aug))

		if self.pct_usage is None:
			pct_usage_display = self.pct_usage
		else:
			pct_usage_display = int(100*self.pct_usage)
		if self.small_prop is None:
			small_prop_display = self.small_prop
		else:
			small_prop_display = int(100*self.small_prop)
		context_aug_filepath = ('../DownloadedData/{}/context_aug'
			'/{}-{}-{}-{}-{}.pickle'.format(
				self.data_name, pct_usage_display, self.small_label, 
				small_prop_display, self.seed, self.split_num
			)
		)
		with open(context_aug_filepath, 'wb') as f:
			pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def create_sst_files(seed):
	small_label = None
	small_prop = None
	lr = 5e-5
	data_name = 'sst'
	for pct_usage in np.arange(0.1, 1.0, 0.1):
		pct_usage = round(pct_usage, 1)
		print(data_name, small_label, small_prop)
		agent = BertAgent(lr, data_name, seed, pct_usage, 
					 	  small_label, small_prop)
		agent.train()
		agent.augment()

def create_subj_files(split_num):
	pct_usage = None
	lr = 5e-5
	seed = 0
	data_name = 'subj'
	# for small_label in [0, 1]:
	for small_label in [1]:
		L = np.arange(0.2, 1.0, 0.1)
		for small_prop in L:
			small_prop = round(small_prop, 1)
			print(data_name, small_label, small_prop)
			agent = BertAgent(lr, data_name, seed, pct_usage, 
						 	  small_label, small_prop, split_num=split_num)
			agent.train()
			agent.augment()

if __name__ == "__main__":
	# try:
	# 	pool = mp.Pool(mp.cpu_count())
	# 	pool.map(create_sst_files, [0, 1, 2, 3])
	# finally:
	# 	pool.close()
	# 	pool.join()
	create_sst_files(4)