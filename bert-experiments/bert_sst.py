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

from agents.bert import BertAgent
from utils.logger import get_bert_logger

import argparse
import json

from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--start_num', type=int, required=True)
    parser.add_argument('-b', '--end_num', type=int, required=True)
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-p', '--pct_usage', type=float)
    parser.add_argument('-r', '--small_prop', type=float)
    parser.add_argument('-l', '--small_label', type=int)
    parser.add_argument('-m', '--aug_mode', type=str)
    parser.add_argument('-u', '--undersample', type=int)
    parser.add_argument('-q', '--geo', type=float)
    arg_dict = vars(parser.parse_args())
    arg_dict['gpu'] = 'cuda:'+str(arg_dict['gpu'])
    if arg_dict['undersample'] is not None:
    	arg_dict['undersample'] = bool(arg_dict['undersample'])
    assert arg_dict['aug_mode'] is not None
    return arg_dict

def experiment(balance_seed):
    kwargs = arg_dict.copy()
    aug_mode = kwargs.pop('aug_mode')
    logger = get_bert_logger(this_script_name, balance_seed, aug_mode=aug_mode)
    device = kwargs.pop('gpu')
    if arg_dict['geo'] is None:
        aug_mode = None
    kwargs['balance_seed'] = balance_seed
    agent = BertAgent(device, logger, data_name, 25, num_epochs, 
    				  lr, aug_mode, 'dev', batch_size, accumulation_steps,
    				  **kwargs)
    agent.run()


arg_dict = get_args()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 4
data_name = 'sst'
batch_size = 32
accumulation_steps = 1
lr = 2e-5

experiment_num_list = list(range(arg_dict.pop('start_num'), arg_dict.pop('end_num')))
try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, experiment_num_list)
finally:
	pool.close()
	pool.join()

# experiment(experiment_num_list[0])
