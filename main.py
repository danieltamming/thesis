import os
import multiprocessing as mp

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device
'''
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


'''
device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr = 0.001
small_label = 0
balance_seed = 0
small_prop = 0.1
geo = 0.2
logger = initialize_logger(this_script_name, balance_seed, other=small_label)
agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
				 'synonym', 'dev', 128, 
				 small_label=small_label, 
				 small_prop=small_prop, 
				 balance_seed=balance_seed,
				 geo=geo)
agent.run()

exit()


device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
def experiment(balance_seed, small_label):
	logger = initialize_logger(this_script_name, balance_seed, other=small_label)
	for small_prop in [0.1, 0.5, 0.9]:
		small_prop = round(small_prop, 2)
		for lr in [i*10**-4 for i in range(2, 10, 2)]:
			for undersample in [False, True]:
				agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
								 None, 'dev', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed, 
								 undersample=undersample)
				agent.run()

print('Number of cpus: {}'.format(mp.cpu_count()))
pool = mp.Pool(mp.cpu_count())
arguments = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), 
			 (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
pool.starmap(experiment, arguments)
pool.close()