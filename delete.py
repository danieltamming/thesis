import numpy as np
import spacy
from transformers import BertTokenizer

from utils.data import get_subj, get_sst

for split_num in range(12):
	data_dict = get_subj(25, 'synonym', split_num=split_num)
# data_dict = get_sst(25, 'synonym')
# print(data_dict.keys())
	for key, data in data_dict.items():
		print(key)
		print(len([1 for tup in data if tup[0] == 0]), 
			  len([1 for tup in data if tup[0] == 1]))
	print(100*'-')