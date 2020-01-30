import os, sys, inspect
current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent

from utils.logger import initialize_logger

initialize_logger(name='exp_trans_subj')
num_epochs = 100
small_label = 1
for balance_seed in range(5):
	for undersample in [False, True]:
		agent = RnnAgent('foo', 'subj', 25, num_epochs, None, 'dev', 128, 
						 small_label=small_label, small_prop=0.5, 
						 balance_seed=balance_seed, undersample=undersample)
		agent.run()
	for geo in np.arange(0.3, 1.0, 0.1):
		geo = round(geo, 2)
		agent = RnnAgent('foo', 'subj', 25, num_epochs, 'trans', 'dev', 128, 
						 small_label=small_label, small_prop=0.5, 
						 balance_seed=balance_seed, geo=geo)
		agent.run()