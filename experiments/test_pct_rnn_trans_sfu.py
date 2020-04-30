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
input_length = 128

param_map = {
	0.1: {'aug': (0.5, 94), 'no': 68},
	0.2: {'aug': (0.7, 48), 'no': 68}, 
	0.3: {'aug': (0.7, 55), 'no': 63}, 
	0.4: {'aug': (0.7, 77), 'no': 45}, 
	0.5: {'aug': (0.7, 43), 'no': 58}, 
	0.6: {'aug': (0.7, 79), 'no': 41}, 
	0.7: {'aug': (0.7, 33), 'no': 70}, 
	0.8: {'aug': (0.7, 35), 'no': 45}, 
	0.9: {'aug': (0.7, 53), 'no': 30},
	1.0: {'aug': (0.7, 81), 'no': 25} 
}

def experiment(balance_seed, split_num):
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	# for pct_usage in np.arange(0.1, 1.1, 0.1):
	for pct_usage in [1.0]:
		pct_usage = round(pct_usage, 2)
		param_pct_map = param_map[pct_usage]

		num_epochs = param_pct_map['no']
		agent = RnnAgent(device, logger, 'sfu', input_length, num_epochs, lr,
						 None, 'dev', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed,
						 split_num=split_num)
		agent.run()
		geo, num_epochs = param_pct_map['aug']
		agent = RnnAgent(device, logger, 'sfu', input_length, num_epochs, lr,
						 'trans', 'dev', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed, 
						 split_num=split_num,
						 geo=geo)
		agent.run()

try:
	split_num_list = list(range(10))
	seed_list = [0, 1]
	params = list(itertools.product(seed_list, split_num_list))
	pool = mp.Pool(mp.cpu_count())
	pool.starmap(experiment, params)
finally:
	pool.close()
	pool.join()

# experiment(0, 0)
