import numpy as np
import spacy
from torch.utils.data import Dataset
from transformers import BertTokenizer

from augs.synonym import syn_aug

class BertDataset(Dataset):
	def __init__(self, data, aug_mode=None, geo=None):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.data = data
		self.aug_mode = aug_mode
		self.geo = geo

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, example = self.data[idx]
		if self.aug_mode is None:
			return self.tokenizer.encode(example), label
		elif self.aug_mode is 'synonym':
			example = syn_aug(example, self.geo)
			return example, label
			# SWITCH COMMENTING -------------------------------------------------
			# return self.tokenizer.encode(example), label
		else:
			raise ValueError('Unrecognized augmentation technique.')

class RnnDataset(Dataset):
	def __init__(self, data, nlp, aug_mode, geo=None):
		# nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
		# ----------------------------
		# nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])
		# nlp.vocab.set_vector(0, vector=np.zeros(nlp.vocab.vectors.shape[1]))
		self.key2row = nlp.vocab.vectors.key2row
		self.nlp = nlp

		self.data = data
		self.aug_mode = aug_mode
		self.geo = geo

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, example = self.data[idx]
		if self.aug_mode is None:
			return self._to_rows(example), label
		elif self.aug_mode is 'synonym':
			example = syn_aug(example, self.geo)
			return self._to_rows(example), label
		else:
			raise ValueError('Unrecognized augmentation technique.')

	def _to_rows(self, example):
		return [self.key2row[token.orth] if token.has_vector 
				else self.key2row[0] for token in self.nlp(example)]