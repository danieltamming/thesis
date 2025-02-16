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
from opt_params import subj_params

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr  = 0.001

# aug_mode = 'synonym'
# aug_mode = 'trans'
aug_mode = 'context'

param_map = subj_params['bal'][aug_mode]

def experiment(balance_seed, split_num):
	logger = initialize_logger(
		this_script_name, balance_seed, other=split_num)
	for small_prop in np.arange(0.1, 1.0, 0.1):
		small_prop = round(small_prop, 2)
		param_prop_map = param_map[small_prop]
		for small_label in [0, 1]:
			geo, num_epochs = param_prop_map['aug']
			geo = round(geo, 2)
			agent = RnnAgent(device, logger, 'subj', 25, num_epochs+1, lr,
							 'synonym', 'test', 128, 
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
								 None, 'test', 128, 
								 small_label=small_label, 
								 small_prop=small_prop, 
								 balance_seed=balance_seed,
								 split_num=split_num, 
								 undersample=undersample)
				agent.run()


# try:
# 	# split_num_list = list(range(10))
# 	# seed_list = [3]
# 	# params = list(itertools.product(seed_list, split_num_list))
# 	missing = [(0,2), (0,3)]
# 	params = missing
# 	pool = mp.Pool(mp.cpu_count())
# 	pool.starmap(experiment, params)
# finally:
# 	pool.close()
# 	pool.join()

experiment(0, 0)