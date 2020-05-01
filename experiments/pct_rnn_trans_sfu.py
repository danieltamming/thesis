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
lr  = 0.01
input_length = 128
def experiment(balance_seed, split_num):
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for pct_usage in np.arange(0.2, 1.1, 0.2):
	# for pct_usage in [1.0]:
		pct_usage = round(pct_usage, 2)
		agent = RnnAgent(device, logger, 'sfu', input_length, num_epochs, lr,
						 None, 'dev', 128, 
						 pct_usage=pct_usage, 
						 balance_seed=balance_seed,
						 split_num=split_num)
		agent.run()
		for geo in np.arange(0.1, 1.0, 0.1):
			geo = round(geo, 2)
			agent = RnnAgent(device, logger, 'sfu', input_length, num_epochs, lr,
							 'trans', 'dev', 128, 
							 pct_usage=pct_usage, 
							 balance_seed=balance_seed, 
							 split_num=split_num,
							 geo=geo)
			agent.run()

try:
	split_num_list = list(range(10))
	# seed_list = list(range(2))
	seed_list = [1]
	params = list(itertools.product(seed_list, split_num_list))
	pool = mp.Pool(mp.cpu_count())
	pool.starmap(experiment, params)
finally:
	pool.close()
	pool.join()

# experiment(0, 0)
