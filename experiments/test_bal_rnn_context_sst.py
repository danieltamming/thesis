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

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr  = 0.001

param_map = {
	0.1: {'aug': (0.3, 1), 'under': 31, 'over': 1},
	0.2: {'aug': (0.3, 3), 'under': 41, 'over': 9}, 
	0.3: {'aug': (0.4, 22), 'under': 30, 'over': 17}, 
	0.4: {'aug': (0.4, 74), 'under': 86, 'over': 10}, 
	0.5: {'aug': (0.4, 58), 'under': 33, 'over': 74}, 
	0.6: {'aug': (0.4, 33), 'under': 27, 'over': 12}, 
	0.7: {'aug': (0.4, 36), 'under': 59, 'over': 88}, 
	0.8: {'aug': (0.4, 33), 'under': 98, 'over': 56}, 
	0.9: {'aug': (0.9, 78), 'under': 35, 'over': 75} 
}

def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		param_prop_map = param_map[small_prop]
		for small_label in [0, 1]:
			geo, num_epochs = param_prop_map['aug']
			agent = RnnAgent(device, logger, 'sst', 25, num_epochs+1, lr,
							 'context', 'test', 128, 
							 small_label=small_label, 
							 small_prop=small_prop, 
							 balance_seed=balance_seed, 
							 geo=geo)
			agent.run()
			for undersample in [False, True]:
				if undersample:
					num_epochs = param_prop_map['under']
				else:
					num_epochs = param_prop_map['over']
				agent = RnnAgent(device, logger, 'sst', 25, num_epochs+1, lr, 
								 None, 'test', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed, 
								 undersample=undersample)
				agent.run()

# experiment(0)
print('Number of cpus: {}'.format(mp.cpu_count()))
try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, list(range(10)))
finally:
	pool.close()
	pool.join()