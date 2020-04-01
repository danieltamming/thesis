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

# augment map from small_prop to its corresponding optimal geo, num_epochs
pct_geo_map = {
	0.1: (5, 0.6), 
	0.2: 0.6, 
	0.3: 0.7, 
	0.4: 0.7, 
	0.5: 0.7, 
	0.6: 0.7, 
	0.7: 0.8, 
	0.8: 0.8, 
	0.9: 0.9
}
pct_over_map = {
	0.1: 3, 
	0.2: , 
	0.3: , 
	0.4: , 
	0.5: , 
	0.6: , 
	0.7: , 
	0.8: , 
	0.9: 
}
# undersample map from small_prop to its optimal num_epochs
pct_under_map = {
	0.1: 86, 
	0.2: , 
	0.3: , 
	0.4: , 
	0.5: , 
	0.6: , 
	0.7: , 
	0.8: , 
	0.9: 
}


def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		for small_label in [0, 1]:
			geo, num_epochs = pct_geo_map[small_prop]
			agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
							 'synonym', 'test', 128, 
							 small_label=small_label, 
							 small_prop=small_prop, 
							 balance_seed=balance_seed, 
							 geo=geo)
			agent.run()
			for undersample in [False, True]:
				if undersample:
					num_epochs = pct_under_map[small_prop]
				else:
					num_epochs = pct_over_map[small_prop]
				agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr, 
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
	ppol.join()