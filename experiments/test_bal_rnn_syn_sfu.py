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
	0.1: {'aug': (0.2, 19), 'under': 17, 'over': 22},
	0.2: {'aug': (0.1, 23), 'under': 40, 'over': 27}, 
	0.3: {'aug': (0.3, 27), 'under': 28, 'over': 30}, 
	0.4: {'aug': (0.3, 42), 'under': 27, 'over': 21}, 
	0.5: {'aug': (0.5, 83), 'under': 24, 'over': 30}, 
	0.6: {'aug': (0.5, 31), 'under': 40, 'over': 26}, 
	0.7: {'aug': (0.5, 39), 'under': 26, 'over': 32}, 
	0.8: {'aug': (0.7, 37), 'under': 50, 'over': 46}, 
	0.9: {'aug': (0.7, 59), 'under': 32, 'over': 42} 
}

def experiment(balance_seed, split_num):
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		param_prop_map = param_map[small_prop]
		for small_label in [0, 1]:
			for undersample in [False, True]:
				if undersample:
					num_epochs = param_prop_map['under']
				else:
					num_epochs = param_prop_map['over']
				agent = RnnAgent(device, logger, 'sfu', input_length, num_epochs, lr,
								 None, 'dev', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed, 
								 split_num=split_num,
								 undersample=undersample)
				agent.run()
			geo, num_epochs = param_prop_map['aug']
			geo = round(geo, 2)
			agent = RnnAgent(device, logger, 'sfu', input_length, num_epochs, lr,
							 'synonym', 'dev', 128, 
							 small_label=small_label, 
							 small_prop=small_prop, 
							 balance_seed=balance_seed, 
							 split_num=split_num,
							 geo=geo)
			agent.run()

try:
	split_num_list = list(range(10))
	# seed_list = list(range(3))
	seed_list = [2]
	params = list(itertools.product(seed_list, split_num_list))
	pool = mp.Pool(mp.cpu_count())
	pool.starmap(experiment, params)
finally:
	pool.close()
	pool.join()

# experiment(0, 0)