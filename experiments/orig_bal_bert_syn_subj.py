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

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 4
data_name = 'subj'
aug_mode = 'synonym'
batch_size = 32
accumulation_steps = 1
def experiment(balance_seed):
	# for small_prop in np.arange(0.1, 1.0, 0.1):
	for small_prop in [0.5]:
		small_prop = round(small_prop, 2)
		for small_label in [0, 1]:
			for undersample in [False, True]:
				agent = BertAgent(device, logger, data_name, 25, num_epochs, 
								  None, 'dev', batch_size, accumulation_steps,
								  small_label=small_label, small_prop=small_prop, 
								  balance_seed=balance_seed, undersample=undersample)
				agent.run()
			for geo in np.arange(0.1, 1.0, 0.1):
				geo = round(geo, 2)
				agent = BertAgent(device, logger, data_name, 25, num_epochs, 
								  aug_mode, 'dev', batch_size, accumulation_steps,
								  small_label=small_label, small_prop=small_prop, 
								  balance_seed=balance_seed, geo=geo)
				agent.run()

# for balance_seed in range(3):
# 	logger = initialize_logger(this_script_name, balance_seed)
# 	experiment(balance_seed)

print('Number of cpus: {}'.format(mp.cpu_count()))

for i in [0, 3, 6]:
	try:
		pool = mp.Pool(mp.cpu_count())
		pool.map(experiment, [i+0, i+1, i+2])
	finally:
		pool.close()
		pool.join()