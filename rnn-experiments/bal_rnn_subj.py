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
lr  = 0.001

# aug_mode = 'synonym'
# aug_mode = 'trans'
aug_mode = 'context'

def experiment(balance_seed, split_num):
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		for small_label in [0, 1]:
			for undersample in [False, True]:
				agent = RnnAgent(device, logger, 'subj', 25, num_epochs, lr,
								 None, 'dev', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed,
								 split_num=split_num, 
								 undersample=undersample)
				agent.run()
			for geo in np.arange(0.1, 1.0, 0.1):
				geo = round(geo, 2)
				agent = RnnAgent(device, logger, 'subj', 25, num_epochs, lr,
								 aug_mode, 'dev', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed, 
								 split_num=split_num,
								 geo=geo)
				agent.run()

# try:
# 	split_num_list = list(range(10))
# 	seed_list = list(range(3))
# 	# seed_list = [3]
# 	params = list(itertools.product(seed_list, split_num_list))
# 	pool = mp.Pool(mp.cpu_count())
# 	pool.starmap(experiment, params)
# finally:
# 	pool.close()
# 	pool.join()

experiment(0, 0)