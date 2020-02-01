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
num_epochs = 10
data_name = 'subj'
aug_mode = 'trans'
batch_size = 32
def experiment(balance_seed):
	for small_label in [0, 1]:
		for undersample in [False, True]:
			agent = BertAgent('foo', logger, data_name, 25, num_epochs, 
							  None, 'dev', batch_size, 
							  small_label=small_label, small_prop=0.5, 
							  balance_seed=balance_seed, undersample=undersample)
			agent.run()
		for geo in np.arange(0.3, 1.0, 0.1):
			geo = round(geo, 2)
			agent = BertAgent('foo', logger, data_name, 25, num_epochs, 
							  aug_mode, 'dev', batch_size, 
							  small_label=small_label, small_prop=0.5, 
							  balance_seed=balance_seed, geo=geo)
			agent.run()

for balance_seed in range(5):
	logger = initialize_logger(this_script_name, balance_seed)
	experiment(balance_seed)