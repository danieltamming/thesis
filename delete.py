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

thing = RnnAgent('foo', 'sst', 25, 100, None, 'dev', 128, 0, 0.5)
# thing = BertAgent('foo', 'sst', 25, 4, 'synonym', 'dev', 32)
thing.run()