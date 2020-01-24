import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from augs.synonym import syn_aug

class BertDataset(Dataset):
	def __init__(self, data, input_length, aug_mode=None, geo=None, translator=None):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.data = data
		self.input_length = input_length
		self.aug_mode = aug_mode
		self.geo = geo
		self.translator = translator

		assert self.aug_mode != 'trans' or self.translator is not None, 'Need to pass in translator model.'

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, example = self.data[idx]
		if self.aug_mode == 'synonym':
			example = syn_aug(example, self.geo)
		elif self.aug_mode == 'trans':
			example = self.translator.aug(example, 'de')
		elif self.aug_mode is not None:
			raise ValueError('Unrecognized augmentation technique.')
		return self._to_tokens(example), label

	def _to_tokens(self, example):
		tknzd = self.tokenizer.encode(
			example, add_special_tokens=True, max_length=self.input_length)
		if len(tknzd) < self.input_length:
			tknzd += (self.input_length - len(tknzd))*[self.tokenizer.pad_token_id]
		return torch.tensor(tknzd, dtype=torch.long)

class RnnDataset(Dataset):
	def __init__(self, data, input_length, nlp, aug_mode, geo=None):
		self.data = data
		self.input_length = input_length
		self.nlp = nlp
		self.key2row = nlp.vocab.vectors.key2row
		self.aug_mode = aug_mode
		self.geo = geo

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, example = self.data[idx]
		if self.aug_mode == 'synonym':
			example = syn_aug(example, self.geo)
		elif self.aug_mode is not None:
			raise ValueError('Unrecognized augmentation technique.')
		return self._to_rows(example), label

	def _to_rows(self, example):
		# also ensures length == self.INPUT_LENGTH
		rows = [self.key2row[token.orth] if token.has_vector
				else self.key2row[0] for token in self.nlp(example)]
		if len(rows) < self.input_length:
			rows += (self.input_length - len(rows))*[self.key2row[0]]
		rows = rows[:self.input_length]
		return torch.tensor(rows, dtype=torch.long)