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
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr  = 0.01

# param_map = {
# 	0.1: {'aug': (0.2, 58), 'under': 51, 'over': 1},
# 	0.2: {'aug': (0.2, 71), 'under': 37, 'over': 7}, 
# 	0.3: {'aug': (0.3, 89), 'under': 39, 'over': 88}, 
# 	0.4: {'aug': (0.3, 92), 'under': 90, 'over': 47}, 
# 	0.5: {'aug': (0.3, 79), 'under': 77, 'over': 85}, 
# 	0.6: {'aug': (0.4, 83), 'under': 88, 'over': 67}, 
# 	0.7: {'aug': (0.7, 93), 'under': 81, 'over': 80}, 
# 	0.8: {'aug': (0.7, 70), 'under': 97, 'over': 60}, 
# 	0.9: {'aug': (0.7, 53), 'under': 62, 'over': 92} 
# }

param_map = {
	0.2: {'aug': (0.1, 85), 'under': 10, 'over': 37}, 
	0.4: {'aug': (0.5, 49), 'under': 54, 'over': 55}, 
	0.6: {'aug': (0.4, 88), 'under': 68, 'over': 15}, 
	0.8: {'aug': (0.8, 69), 'under': 73, 'over': 84}, 
}

def experiment(split_num):
	balance_seed = 0
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for small_prop in np.arange(0.2, 1.0, 0.2):
		small_prop = round(small_prop, 2)
		param_prop_map = param_map[small_prop]
		for small_label in [0, 1]:
			geo, num_epochs = param_prop_map['aug']
			agent = RnnAgent(device, logger, 'subj', 25, num_epochs+1, lr,
							 'context', 'dev', 128, 
							 small_label=small_label, 
							 small_prop=small_prop, 
							 balance_seed=balance_seed,
							 split_num=split_num,
							 geo=geo)
			agent.run()
			for undersample in [False, True]:
				if undersample:
					num_epochs = param_prop_map['under']
				else:
					num_epochs = param_prop_map['over']
				agent = RnnAgent(device, logger, 'subj', 25, num_epochs+1, lr,
								 None, 'dev', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed,
								 split_num=split_num, 
								 undersample=undersample)
				agent.run()

try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, list(range(10)))
finally:
	pool.close()
	pool.join()

# experiment(0)