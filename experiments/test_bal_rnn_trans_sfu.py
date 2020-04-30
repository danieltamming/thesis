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
	0.1: {'aug': (0.5, 16), 'under': 11, 'over': 16},
	0.2: {'aug': (0.7, 36), 'under': 19, 'over': 19}, 
	0.3: {'aug': (0.7, 42), 'under': 32, 'over': 34}, 
	0.4: {'aug': (0.7, 40), 'under': 27, 'over': 24}, 
	0.5: {'aug': (0.7, 30), 'under': 30, 'over': 43}, 
	0.6: {'aug': (0.8, 54), 'under': 78, 'over': 34}, 
	0.7: {'aug': (0.9, 35), 'under': 43, 'over': 67}, 
	0.8: {'aug': (0.9, 37), 'under': 18, 'over': 54}, 
	0.9: {'aug': (0.9, 38), 'under': 38, 'over': 38} 
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
							 'trans', 'dev', 128, 
							 small_label=small_label, 
							 small_prop=small_prop, 
							 balance_seed=balance_seed, 
							 split_num=split_num,
							 geo=geo)
			agent.run()

try:
	split_num_list = list(range(10))
	# seed_list = list(range(3))
	seed_list = [0, 1]
	params = list(itertools.product(seed_list, split_num_list))
	pool = mp.Pool(mp.cpu_count())
	pool.starmap(experiment, params)
finally:
	pool.close()
	pool.join()

# experiment(0, 0)
