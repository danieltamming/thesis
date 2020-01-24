from tqdm import tqdm
import time

import numpy as np
import spacy
import torch
import torch.nn as nn

from managers.trec import TrecDatasetManager, get_trec
from managers.sst import get_sst
from managers.subj import get_subj
from agents.rnn import RnnAgent
from agents.bert import BertAgent
from augs.synonym import syn_aug

# with open('../DownloadedData/sst/train.txt') as f:
# 	for line in tqdm(f.read().splitlines()):
# 		example = line.split(maxsplit=1)[1].lstrip()
# 		syn_aug(example, 0.5)
# exit()

# for data_name in ['sst', 'trec', 'subj']:
# 	for aug_mode in ['synonym', None]:
# 		print('\nRNN', data_name, aug_mode)
# 		thing = RnnAgent('foo', data_name, 25, 1, aug_mode, 'dev', 128)
# 		thing.run()
# 		print('\nBERT', data_name, aug_mode)
# 		thing = BertAgent('foo', data_name, 25, 1, aug_mode, 'dev', 32)
# 		thing.run()
# exit()

# thing = RnnAgent('foo', 'sst', 25, 100, 'trans', 'dev', 128)
thing = BertAgent('foo', 'sst', 25, 4, 'trans', 'dev', 32)
thing.run()



# nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])
# nlp.vocab.set_vector(0, vector=np.zeros(nlp.vocab.vectors.shape[1]))
# key2row = nlp.vocab.vectors.key2row
# emb = nn.Embedding.from_pretrained(torch.from_numpy(nlp.vocab.vectors.data))

non_auged = [example for _, example in get_trec()['train']]

sentence = 'What asdafsdfsd adfasdfsdaa stole the cork from my lunch?'
doc = nlp(sentence)
rows = torch.tensor(
	[key2row[token.orth] if token.has_vector else key2row[0] for token in doc], 
	dtype=torch.long)
vec = emb(rows)

v1 = vec.numpy()
v2 = np.stack([token.vector for token in doc])
assert np.array_equal(v1, v2)

exit()
rows = torch.from_numpy(np.vectorize(key2row.get)(orths))
print(rows)
exit()
# print(example.to_array('ID'))
# print([word.orth for word in example])
print(len(nlp.vocab.vectors.key2row))

word = example[5]
print(word)
row = key2row[word.orth]
print(row)
vocab_row = torch.tensor(row, dtype=torch.long)
vec = emb(vocab_row)

v1 = vec.numpy()
v2 = word.vector
print(np.array_equal(v1, v2))

exit()

words = set([word for example in non_auged for word in example.split()])
print(len(words))
print(sum([nlp(word)[0].is_oov for word in words]))
exit()

sentence = non_auged[0]
print(sentence)
doc = nlp(sentence)
print([word.has_vector for word in doc])

exit()

# -------------------------------------------------------------------
'''
thing = TrecDatasetManager('foo', 'rnn', None, 1, 0.5)
dataset = thing.get_train_set()

non_auged = [example for _, example in get_trec()['train']]
print(max([len(example.split()) for example in non_auged]))

auged = []
for example, label in tqdm(dataset):
	auged.append(example)
	# auged.append(example)
	# print(example)
	# print(' '.join([token.text for token in example]) == non_auged[0])
	# exit()
exit()

num_same = 0
for s1, s2 in zip(auged, non_auged):
	num_same += s1 == s2
print(num_same/len(auged))

exit()
'''

from managers.subj import SubjDatasetManager, get_subj

non_auged = [example for _, example in get_subj()]
print(max([len(example.split()) for example in non_auged]))
exit()

thing = SubjDatasetManager('foo', 'bert', None, 1, 0.5)
dataset = thing.get_train_set()

auged = []
for example, label in tqdm(dataset):
	auged.append(example)

num_same = 0
for s1, s2 in zip(auged, non_auged):
	num_same += s1 == s2
print(num_same/len(auged))

exit()

from managers.sst import SSTDatasetManager, get_sst

thing = SSTDatasetManager('foo', 'bert', None, 1, 0.5)
dataset = thing.get_train_set()
non_auged = [example for _, example in get_sst()['train']]

auged = []
for example, label in tqdm(dataset):
	auged.append(example)

num_same = 0
for s1, s2 in zip(auged, non_auged):
	num_same += s1 == s2
print(num_same/len(auged))