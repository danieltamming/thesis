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
from utils.logger import get_bert_logger
# from utils.parsing import get_device

import argparse
import json

from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--start_seed', type=int, required=True)
    parser.add_argument('-b', '--end_seed', type=int, required=True)
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-p', '--pct_usage', type=float, required=True)
    parser.add_argument('-u', '--undersample', type=int)
    parser.add_argument('-q', '--geo', type=float)
    arg_dict = vars(parser.parse_args())
    arg_dict['gpu'] = 'cuda:'+str(arg_dict['gpu'])
    if arg_dict['undersample'] is None:
    	del arg_dict['undersample']
    else:
    	arg_dict['undersample'] = arg_dict['undersample'] == 1
    if arg_dict['geo'] is None:
    	del arg_dict['geo']
    	arg_dict['aug_mode'] = None
    else:
    	arg_dict['aug_mode'] = 'trans'
    return arg_dict

def experiment(balance_seed):
	logger = get_bert_logger(this_script_name, balance_seed)
	kwargs = arg_dict.copy()
	device = kwargs.pop('gpu')
	aug_mode = kwargs.pop('aug_mode')
	kwargs['balance_seed'] = balance_seed
	agent = BertAgent(device, logger, data_name, 25, num_epochs, 
					  lr, aug_mode, 'dev', batch_size, accumulation_steps,
					  **kwargs)
	agent.run()


arg_dict = get_args()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 4
data_name = 'sst'
aug_mode = 'trans'
batch_size = 32
accumulation_steps = 1
lr = 2e-5

seed_list = list(range(arg_dict.pop('start_seed'), arg_dict.pop('end_seed')))
try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, seed_list)
finally:
	pool.close()
	pool.join()

# experiment(0)