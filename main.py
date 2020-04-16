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
import torch

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 3
data_name = 'sst'
aug_mode = None
batch_size = 32
accumulation_steps = 1
input_length = 512
def experiment(balance_seed):
	logger = initialize_logger(this_script_name, balance_seed)
	agent = BertAgent(device, logger, data_name, input_length, num_epochs, 
					  aug_mode, 'dev', batch_size, accumulation_steps,
					  pct_usage=1,
					  balance_seed=balance_seed)
	agent.run()

# try:
# 	pool = mp.Pool(mp.cpu_count())
# 	pool.map(experiment, [0, 1, 2, 3])
# finally:
# 	pool.close()
# 	pool.join()

experiment(0)