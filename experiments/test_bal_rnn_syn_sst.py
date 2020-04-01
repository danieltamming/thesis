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

# map from small_prop to its corresponding optimal geo
pct_geo_map = {
	0.1: 6, 
	0.2: 6, 
	0.3: 7, 
	0.4: 7, 
	0.5: 7, 
	0.6: 7, 
	0.7: 8, 
	0.8: 8, 
	0.9: 9
}

def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		for small_label in [0, 1]:
			for undersample in [False, True]:
				agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr, 
								 None, 'dev', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed, 
								 undersample=undersample)
				agent.run()
			geo = pct_geo_map[small_prop]
			agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
							 'synonym', 'dev', 128, 
							 small_label=small_label, 
							 small_prop=small_prop, 
							 balance_seed=balance_seed, 
							 geo=geo)
			agent.run()

print('Number of cpus: {}'.format(mp.cpu_count()))
pool = mp.Pool(mp.cpu_count())
pool.map(experiment, list(range(30)))
pool.close()