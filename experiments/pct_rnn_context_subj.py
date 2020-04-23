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

def experiment(split_num):
	balance_seed = 0
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for pct_usage in np.arange(0.1, 1.1, 0.1):
		pct_usage = round(pct_usage, 2)
		agent = RnnAgent(device, logger, 'subj', 25, num_epochs, lr, 
						 None, 'dev', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed,
						 split_num=split_num)
		agent.run()
		for geo in np.arange(0.1, 1.0, 0.1):
			geo = round(geo, 2)
			agent = RnnAgent(device, logger, 'subj', 25, num_epochs, lr,
							 'context', 'dev', 128, 
							 pct_usage=pct_usage, 
							 balance_seed=balance_seed, 
							 split_num=split_num,
							 geo=geo)
			agent.run()

try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, list(range(10)))
finally:
	pool.close()
	pool.join()
