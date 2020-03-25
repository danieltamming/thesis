import sys
import os
import inspect
import random
from collections import Counter
import string

import numpy as np

import nltk
from nltk.corpus import wordnet as wn

current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.data import get_sst, get_subj, get_trec

def get_wordnet_pos(tag):
	tag_dict = {'J': wn.ADJ, 'N': wn.NOUN, 'V': wn.VERB, 'R':wn.ADV}
	return tag_dict.get(tag[0], None)

def get_synonyms(word, tag, min_reputation):
	if len(word) <= 2:
		return []
	synonyms = Counter()
	for syn in wn.synsets(word, get_wordnet_pos(tag)):
		for lemma in syn.lemmas():
			synonym = lemma.name().replace('_',' ').replace('-',' ')
			if synonym != word: 
				synonyms.update({synonym:lemma.count()})
	synonyms = Counter({synonym:synonyms[synonym] for synonym in synonyms if synonyms[synonym] >= min_reputation})
	return [word.split() for word,_ in synonyms.most_common()]

def get_synonym_dict(seq, min_reputation):
	idx_to_syns = {}
	for i, (word, tag) in enumerate(nltk.pos_tag(seq)):
		synonyms = get_synonyms(word, tag, min_reputation)
		if synonyms:
			idx_to_syns[i] = synonyms
	return idx_to_syns

def syn_aug(example, geo, min_reputation=2):
	# shift so minimum value of geometric distribution is 0
	num_want_replace = np.random.geometric(geo) - 1
	if num_want_replace == 0:
		return example
	seq = example.split()
	syn_dict = get_synonym_dict(seq, min_reputation)
	# can't replace more words than there are synonyms for
	num_to_replace = min(num_want_replace, len(syn_dict))
	if num_want_replace == 0:
		return example
	idxs = random.sample(syn_dict.keys(), num_to_replace)
	new_seq = []
	for i, word in enumerate(seq):
		if i in idxs:
			syn_idx = min(np.random.geometric(geo), len(syn_dict[i])) - 1
			new_seq.extend(syn_dict[i][syn_idx])
		else:
			new_seq.append(word)
	return ' '.join(new_seq)

def create_files(data_name, data_func):
	data_dir = os.path.join('../DownloadedData/', data_name)
	syn_aug_dir = os.path.join(data_dir, 'syn_aug')
	print(syn_aug_dir)



		# with open(syn_aug_filepath, 'wb') as f:
		# 	pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	create_files('sst', get_sst)