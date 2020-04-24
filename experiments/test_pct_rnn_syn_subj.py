import os
import sys
import inspect
import multiprocessing as mp
import itertools

current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np

from agents.rnn import RnnAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr  = 0.001

param_map = {
	0.1: {'aug': (0.4, 95), 'no': 73},
	0.2: {'aug': (0.5, 92), 'no': 90}, 
	0.3: {'aug': (0.3, 94), 'no': 99}, 
	0.4: {'aug': (0.4, 86), 'no': 62}, 
	0.5: {'aug': (0.6, 83), 'no': 76}, 
	0.6: {'aug': (0.4, 91), 'no': 89}, 
	0.7: {'aug': (0.6, 69), 'no': 85}, 
	0.8: {'aug': (0.4, 89), 'no': 89}, 
	0.9: {'aug': (0.3, 95), 'no': 74},
	1.0: {'aug': (0.4, 63), 'no': 65} 
}

def experiment(balance_seed, split_num):
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for pct_usage in np.arange(0.1, 1.1, 0.1):
		pct_usage = round(pct_usage, 2)
		param_pct_map = param_map[pct_usage]

		num_epochs = param_pct_map['no']
		agent = RnnAgent(device, logger, 'subj', 25, num_epochs, lr, 
						 None, 'test', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed,
						 split_num=split_num)
		agent.run()
		
		geo, num_epochs = param_pct_map['aug']
		agent = RnnAgent(device, logger, 'subj', 25, num_epochs, lr,
						 'syn', 'test', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed, 
						 split_num=split_num,
						 geo=geo)
		agent.run()

try:
	split_num_list = list(range(10))
	seed_list = list(range(3))
	params = list(itertools.product(seed_list, split_num_list))
	pool = mp.Pool(mp.cpu_count())
	pool.starmap(experiment, params)
finally:
	pool.close()
	pool.join()

# experiment(0, 0)