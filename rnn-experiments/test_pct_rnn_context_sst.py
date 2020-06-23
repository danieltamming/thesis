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
	0.1: {'aug': (0.6, 19), 'no': 15},
	0.2: {'aug': (0.1, 22), 'no': 98}, 
	0.3: {'aug': (0.2, 37), 'no': 59}, 
	0.4: {'aug': (0.7, 94), 'no': 87}, 
	0.5: {'aug': (0.1, 39), 'no': 74}, 
	0.6: {'aug': (0.1, 39), 'no': 91}, 
	0.7: {'aug': (0.1, 20), 'no': 21}, 
	0.8: {'aug': (0.1, 32), 'no': 98}, 
	0.9: {'aug': (0.1, 21), 'no': 52},
	1.0: {'aug': (0.1, 33), 'no': 50} 
}

def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for pct_usage in np.arange(0.1, 1.1, 0.1):
		pct_usage = round(pct_usage, 2)
		param_pct_map = param_map[pct_usage]

		geo, num_epochs = param_pct_map['aug']
		agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
						 'context', 'test', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed, 
						 geo=geo)
		agent.run()

		num_epochs = param_pct_map['no']
		agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr, 
						 None, 'test', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed)
		agent.run()

print('Number of cpus: {}'.format(mp.cpu_count()))
try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, list(range(10)))
finally:
	pool.close()
	pool.join()

# experiment(0)