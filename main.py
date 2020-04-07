import os
import multiprocessing as mp

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_device

# USE SPLIT_NUM PARAMETER FOR CROSSTEST

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

device = get_device()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
lr  = 0.001
input_length = 80
logger = initialize_logger(this_script_name, 0, other=0)
split_num = 9
agent = RnnAgent(device, logger, 'subj', input_length, num_epochs, lr,
				 None, 'dev', 128, 
				 small_label=0, 
				 small_prop=0.5, 
				 balance_seed=0,
				 split_num=split_num)
agent.run()