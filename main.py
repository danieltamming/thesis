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
num_epochs = 4
lr = 0.001
accumulation_steps = 1
pct_usage = 0.5
balance_seed = 0
geo = 0.5
batch_size = 16
split_num = 9
logger = initialize_logger(this_script_name, balance_seed, other=pct_usage)
agent = BertAgent(device, logger, 'subj', 25, num_epochs,
				 'trans', 'test', batch_size, accumulation_steps,
				 pct_usage=pct_usage, 
				 balance_seed=balance_seed,
				 split_num=split_num,
				 verbose=True
				 )
agent.run()
exit()

for split_num in range(10):
	agent = BertAgent(device, logger, 'subj', 25, num_epochs,
					 'trans', 'test', batch_size, accumulation_steps,
					 pct_usage=pct_usage, 
					 balance_seed=balance_seed,
					 geo=geo,
					 split_num=split_num)
	agent.run()

	agent = BertAgent(device, logger, 'subj', 25, num_epochs,
					 None, 'test', batch_size, accumulation_steps,
					 pct_usage=pct_usage, 
					 balance_seed=balance_seed,
					 split_num=split_num)
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