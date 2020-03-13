import random
from itertools import cycle, islice

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils.data import partition_within_classes, get_sst, get_subj
from augs.synonym import syn_aug
from augs.trans import trans_aug
from augs.context import context_aug

class DatasetBase(Dataset):
	def __init__(self, to_ids_func):
		self.to_ids_func = to_ids_func

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, example, aug_dict = self.data[idx]
		# if use case is dataset balancing but this is not small label
		if self.small_label is not None and self.small_label != label:
			pass
		elif self.aug_mode == 'synonym':
			example = syn_aug(example, self.geo)
		elif self.aug_mode == 'trans':
			example = trans_aug(example, aug_dict, self.geo)
		elif self.aug_mode == 'context':
			example = context_aug(example, aug_dict, self.geo)
			# wasting time here when model is bert by 
			# converting to string then back to ids
		elif self.aug_mode is not None:
			raise ValueError('Unrecognized augmentation technique.')

		if self.aug_mode == 'context':
			example = self.tokenizer.decode(example, skip_special_tokens=True)

		# print(label, example)

		return self.to_ids_func(example), label

	def _im_re_balance(self, data, balance_seed, undersample):
		'''
		Remove small_prop proportion of data with label small_label
		Then either duplicate this data, so it will be artificially rebalanced
		With augmentation, or undersample the data.

		TODO check it is fit to cases with more than 2 classes
		'''
		random.seed(balance_seed)
		other_data = [tup for tup in data if tup[0] != self.small_label]
		label_data = [tup for tup in data if tup[0] == self.small_label]
		print(len(other_data), len(label_data))
		num_orig = len(label_data)
		num_keep = int(self.small_prop*num_orig)
		label_data = random.sample(label_data, num_keep)
		print(len(other_data), len(label_data))
		if undersample:
			other_data = self._undersample(other_data, num_keep)
		else:
			label_data = list(islice(cycle(label_data), num_orig))
		print(len(other_data), len(label_data))
		return other_data + label_data

	def _undersample(self, other_data, num_keep):
		label_dict = {}
		for tup in other_data:
			label = tup[0]
			if label in label_dict:
				label_dict[label].append(tup)
			else:
				label_dict[label] = [tup]
		res_data = []
		for data_list in label_dict.values():
			if len(data_list) <= num_keep:
				res_data.extend(data_list)
			else:
				res_data.extend(random.sample(data_list, num_keep))
		return res_data


class BertDataset(DatasetBase):
	def __init__(self, data, input_length, aug_mode, pct_usage=None, geo=None,
				 small_label=None, small_prop=None, balance_seed=None,
				 undersample=False, tokenizer=None):
		super().__init__(self._to_tokens)
		self.input_length = input_length
		self.aug_mode = aug_mode
		self.pct_usage = pct_usage
		self.geo = geo
		self.small_label = small_label
		self.small_prop = small_prop
		self.undersample = undersample

		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.tokenizer = tokenizer

		if (small_label is not None and small_prop is not None and 
				aug_mode != 'context'):
			self.data = self._im_re_balance(data, balance_seed, undersample)
		elif pct_usage is not None:
			self.data, _ = partition_within_classes(
				data, pct_usage, True, seed=balance_seed)
		else:
			self.data = data

		##############################################
		# TEST ABOVE LOGIC IN HERE AND RNNDATASET
		##############################################

	def _to_tokens(self, example):
		tknzd = self.tokenizer.encode(
			example, add_special_tokens=True, max_length=self.input_length)
		if len(tknzd) < self.input_length:
			tknzd += (self.input_length - len(tknzd))*[self.tokenizer.pad_token_id]
		return torch.tensor(tknzd, dtype=torch.long)


class RnnDataset(DatasetBase):
	def __init__(self, nlp, data, input_length, aug_mode, pct_usage=None, geo=None, 
				 small_label=None, small_prop=None, balance_seed=None,
				 undersample=False, tokenizer=None):
		super().__init__(self._to_rows)
		self.input_length = input_length
		self.nlp = nlp
		self.key2row = nlp.vocab.vectors.key2row
		self.aug_mode = aug_mode
		self.geo = geo
		self.small_label = small_label
		self.small_prop = small_prop
		self.undersample = undersample

		# if aug_mode == 'context':
		# 	self.tokenizer = BertTokenizer.from_pretrained(
		# 		'bert-base-uncased')
		self.tokenizer = tokenizer

		if (small_label is not None and small_prop is not None and 
				aug_mode != 'context'):
			self.data = self._im_re_balance(data, balance_seed, undersample)
		elif pct_usage is not None:
			self.data, _ = partition_within_classes(
				data, pct_usage, True, seed=balance_seed)
		else:
			self.data = data

	def _to_rows(self, example):
		# also ensures length == self.INPUT_LENGTH
		rows = [self.key2row[token.orth] if token.has_vector
				else self.key2row[0] for token in self.nlp(example)]
		if len(rows) < self.input_length:
			rows += (self.input_length - len(rows))*[self.key2row[0]]
		rows = rows[:self.input_length]
		return torch.tensor(rows, dtype=torch.long)


class TestTimeDataset(BertDataset):
	def __init__(self, data_name, input_length, aug_mode):
		assert aug_mode == 'trans'
		if data_name == 'sst':
			self.data = get_sst(input_length, aug_mode)['dev']
		elif data_name == 'subj':
			self.data = get_subj(input_length, aug_mode)['dev']
		super().__init__(self.data, input_length, aug_mode)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, example, aug_dict = self.data[idx]
		if self.aug_mode == 'synonym':
			raise NotImplementedError('Synonym test time aug.')
			# aug_examples = syn_aug(example, self.geo)
		elif self.aug_mode == 'trans':
			seqs = torch.cat([self.to_ids_func(example).unsqueeze(0)] 
							 + [self.to_ids_func(s).unsqueeze(0) for s 
								in aug_dict.keys()])
			weights = torch.tensor([1] + list(aug_dict.values()))
			return seqs, weights, label
		else:
			raise ValueError('Unrecognized augmentation technique.')