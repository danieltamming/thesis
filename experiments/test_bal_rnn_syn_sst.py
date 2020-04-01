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
	0.1: {'aug': (0.6, 1), 'under': 20, 'over': 1},
	0.2: {'aug': (0.6, 10), 'under': 40, 'over': 7}, 
	0.3: {'aug': (0.7, 7), 'under': 23, 'over': 14}, 
	0.4: {'aug': (0.7, 14), 'under': 63, 'over': 22}, 
	0.5: {'aug': (0.7, 18), 'under': 52, 'over': 40}, 
	0.6: {'aug': (0.7, 44), 'under': 69, 'over': 49}, 
	0.7: {'aug': (0.8, 79), 'under': 92, 'over': 61}, 
	0.8: {'aug': (0.7, 99), 'under': 40, 'over': 99}, 
	0.9: {'aug': (0.8, 91), 'under': 83, 'over': 81} 
}

def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		param_prop_map = param_map[small_prop]
		for small_label in [0, 1]:
			geo, num_epochs = param_prop_map['aug']
			agent = RnnAgent(device, logger, 'sst', 25, num_epochs+1, lr,
							 'synonym', 'test', 128, 
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
	pool.map(experiment, list(range(30)))
finally:
	pool.close()
	pool.join()