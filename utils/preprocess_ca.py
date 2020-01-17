import os
import string
import re
import pickle
import time
from collections import Counter

import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer

def get_word_counter(tokenizer, data, embed_vocab, embed_suffixes):
	count = 0
	word_counter = Counter()
	for (seq, cat, aug) in tqdm(data):
		# combos = [tokenizer.convert_ids_to_tokens([seq[i]] + aug[i]) 
		# 	   for i in range(len(seq))]

		# fix aug indexing (subtract by 1) in aug creation to avoid having to add 1

		combos = [tokenizer.convert_ids_to_tokens([seq[i]] + aug[i]) 
			   for i in range(len(seq))]
		prev_suffixes = []
		for arr in reversed(combos):
			next_suffixes = []
			for tok in arr:
				for suffix in [''] + prev_suffixes:
					total_tok = tok + suffix
					if total_tok in embed_vocab:
						word_counter.update([total_tok])
					elif '#' in total_tok and (total_tok[2:] in embed_suffixes
							or total_tok[:2] in embed_vocab):
						assert total_tok[:2] == '##' and '#' not in total_tok[2:]
						next_suffixes.append(total_tok[2:])
			prev_suffixes = next_suffixes
	return word_counter


def get_embed_vocab(embed_filename):
	embed_vocab = set()
	with open(embed_filename, 'r') as f:
		num_lines, embed_dims = [int(num) for num in f.readline().split()]
		for _ in tqdm(range(num_lines)):
			line = f.readline()
			word, vec = line.split(' ',1)
			embed_vocab.add(word)
	return embed_vocab

def get_suffixes(vocab):
	'''
	Returns all possible suffixes EXCEPT the words themselves
	'''
	suffixes = set()
	for word in tqdm(vocab):
		suffixes.update([word[i:] for i in range(1, len(word))])
	return suffixes

def get_paths():
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	embed_filename = os.path.join(data_parent, 'enwiki_20180420_300d.txt')
	target_path = os.path.join(repo_path, 'data', 'procon_ca')
	return embed_filename, target_path

def save_vocab_embed(embed_filename, target_path, word_counter):
	embeddings = {}
	with open(embed_filename) as f:
		num_lines, embed_dims = [int(num) for num in f.readline().split()]
		embeddings[''] = np.zeros(embed_dims)
		for _ in tqdm(range(num_lines)):
			line = f.readline()
			word, vec = line.split(' ', 1)
			if word in word_counter:
				count += 1
				embeddings[word] = np.array([float(num) for num 
											 in vec.split()])
	embed_target = os.path.join(target_path, 'embeddings.pickle')
	with open(embed_target, 'wb') as f:
		pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

def process():
	data_filename = 'data/procon_ca/aug_bert_ids.pickle'
	with open(data_filename, 'rb') as f:
		data = pickle.load(f)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	embed_filename, target_path = get_paths()
	embed_vocab = get_embed_vocab(embed_filename)
	embed_suffixes = get_suffixes(embed_vocab)
	print(len(embed_vocab))
	print(len(embed_suffixes))
	word_counter = get_word_counter(tokenizer, data, embed_vocab, embed_suffixes)
	print(len(word_counter))
	save_vocab_embed(embed_filename, target_path, word_counter)

if __name__ == "__main__":
	process()