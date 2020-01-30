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

this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100

def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for small_label in [0, 1]:
		for undersample in [False, True]:
			agent = RnnAgent('foo', logger, 'subj', 25, num_epochs, None, 'dev', 128, 
							 small_label=small_label, small_prop=0.5, 
							 balance_seed=balance_seed, undersample=undersample)
			agent.run()
		for geo in np.arange(0.3, 1.0, 0.1):
			geo = round(geo, 2)
			agent = RnnAgent('foo', logger, 'subj', 25, num_epochs, 'synonym', 'dev', 128, 
							 small_label=small_label, small_prop=0.5, 
							 balance_seed=balance_seed, geo=geo)
			agent.run()

print('Number of cpus: {}'.format(mp.cpu_count()))
pool = mp.Pool(mp.cpu_count())
pool.map(experiment, list(range(5)))
pool.close()