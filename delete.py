import numpy as np
import spacy
from transformers import BertTokenizer
from collections import Counter

from utils.data import get_subj, get_sst

for seed in [0]:
	counts = Counter()
	for split_num in range(10):
		test = get_subj(25, None, seed=seed, split_num=split_num)['test']
		counts.update([tup[1] for tup in test])

	all_data = get_subj(25, None, seed=seed, gen_splits=False)
	other_counts = Counter([tup[1] for tup in all_data])
	assert other_counts == counts