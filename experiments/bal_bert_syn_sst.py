import os
import sys
import inspect
import multiprocessing as mp

current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import torch

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device



import gc
from collections import Counter
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



device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 4
data_name = 'sst'
aug_mode = 'synonym'
batch_size = 32
accumulation_steps = 1
def experiment(balance_seed):
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		for small_label in [0, 1]:
			for undersample in [False, True]:

				tensors = get_tensors()
				for key, count in tensors.items():
					print(key, count)
				print(sum(tensors.values()))
				print(torch.cuda.memory_allocated())

				agent = BertAgent(device, logger, data_name, 25, num_epochs, 
								  None, 'dev', batch_size, accumulation_steps,
								  small_label=small_label, small_prop=small_prop, 
								  balance_seed=balance_seed, undersample=undersample)
				agent.run()
				# del agent.model
			for geo in np.arange(0.5, 1.0, 0.1):

				tensors = get_tensors()
				for key, count in tensors.items():
					print(key, count)
				print(sum(tensors.values()))
				print(torch.cuda.memory_allocated())

				geo = round(geo, 2)
				agent = BertAgent(device, logger, data_name, 25, num_epochs, 
								  aug_mode, 'dev', batch_size, accumulation_steps,
								  small_label=small_label, small_prop=small_prop, 
								  balance_seed=balance_seed, geo=geo)
				agent.run()
				# del agent.model

for balance_seed in range(3):
	logger = initialize_logger(this_script_name, balance_seed)
	experiment(balance_seed)