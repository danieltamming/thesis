import os
import multiprocessing as mp

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

# USE SPLIT_NUM PARAMETER FOR CROSSTEST

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr = 0.001
pct_usage = 0.5
balance_seed = 0
# geo = 0.5
logger = initialize_logger(this_script_name, balance_seed, other=pct_usage)
for geo in [0.3, 0.5, 0.7]:
	agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
					 'trans', 'dev', 128, 
					 pct_usage=pct_usage, 
					 balance_seed=balance_seed,
					 geo=geo)
	agent.run()

agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
				 None, 'dev', 128, 
				 pct_usage=pct_usage, 
				 balance_seed=balance_seed)
agent.run()

# device = get_device()
# this_script_name = os.path.basename(__file__).split('.')[0]
# num_epochs = 100
# lr = 0.001
# small_label = 0
# balance_seed = 0
# small_prop = 0.1
# geo = 0.2
# logger = initialize_logger(this_script_name, balance_seed, other=small_label)
# agent = RnnAgent(device, logger, 'sst', 25, num_epochs, lr,
# 				 'synonym', 'dev', 128, 
# 				 small_label=small_label, 
# 				 small_prop=small_prop, 
# 				 balance_seed=balance_seed,
# 				 geo=geo)
# agent.run()