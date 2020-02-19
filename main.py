import os

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 4
batch_size = 32
accumulation_steps = 1
seed = 0
pct_usage = None
small_label = 0
small_prop = 0.1
data_name = 'sst'
# data_name = 'subj'
mode = 'dev'
# mode = 'save'
# mode = 'test-aug'
# aug_mode = 'trans'
aug_mode = 'synonym'
# aug_mode = None

import gc
from collections import Counter
import torch
def get_tensors():
	counts = Counter()
	for obj in gc.get_objects():
	    try:
	        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
	            # print(type(obj), obj.size())
	            counts[tuple(obj.shape)] += 1
	    except:
	        pass
	return counts

tensors = get_tensors()
for key, count in tensors.items():
	print(key, count)
print(sum(tensors.values()))
print(torch.cuda.memory_allocated())

logger = initialize_logger(this_script_name, seed)
thing = BertAgent(device, logger, data_name, 25, num_epochs, 
				  aug_mode, mode, batch_size, accumulation_steps,
				  small_label=small_label, small_prop=small_prop,
				  balance_seed=seed, undersample=False,
				  pct_usage=pct_usage, verbose=False, geo=0.9)
thing.run()

tensors = get_tensors()
for key, count in tensors.items():
	print(key, count)
print(sum(tensors.values()))
print(torch.cuda.memory_allocated())

# batch_size = 64
# num_epochs = 100
# thing = RnnAgent(device, logger, 'sst', 25, num_epochs, 
# 				  aug_mode, mode, batch_size,
# 				  small_label=small_label, small_prop=small_prop,
# 				  balance_seed=seed, undersample=False,
# 				  pct_usage=pct_usage, verbose=True)
thing.run()

exit()

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 10
aug_mode = None
batch_size = 4
accumulation_steps = 4
for balance_seed in range(5):
	logger = initialize_logger(this_script_name, balance_seed)
	thing = BertAgent(device, logger, 'sst', 25, num_epochs, 
					  aug_mode, 'dev', batch_size, accumulation_steps,
					  small_label=None, small_prop=None,
					  balance_seed=balance_seed, undersample=False,
					  pct_usage=1)
	thing.run()