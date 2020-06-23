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
from opt_params import sst_params

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr  = 0.001

# aug_mode = 'synonym'
# aug_mode = 'trans'
aug_mode = 'context'

param_map = sst_params['pct'][aug_mode]

def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	for pct_usage in np.arange(0.1, 1.1, 0.1):
		pct_usage = round(pct_usage, 2)
		param_pct_map = param_map[pct_usage]

		geo, num_epochs = param_pct_map['aug']
		agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
						 'synonym', 'test', 128, 
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

# try:
# 	pool = mp.Pool(mp.cpu_count())
# 	pool.map(experiment, [10, 28, 2, 7])
# finally:
# 	pool.close()
# 	pool.join()

experiment(0)