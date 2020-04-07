import numpy as np
import spacy
from transformers import BertTokenizer
from collections import Counter
import pickle

from utils.data import get_subj, get_sst

for num in range(10):
	train_file = 'delete/split_{}.pickle'.format(num)
	with open(train_file, 'rb') as f:
		train = pickle.load(f)
	val_file = 'delete/test_split_{}.pickle'.format(num)
	with open(val_file, 'rb') as f:
		val = pickle.load(f)

	train_seqs = [seq for _, seq in train]
	val_seqs = [seq for _, seq, _ in val]
	print(len(set(train_seqs)))
	print(len(set(train_seqs).intersection(set(train_seqs))))