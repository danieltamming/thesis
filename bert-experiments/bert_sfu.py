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
    parser.add_argument('-o', '--only_original', action='store_true', default=False)
    parser.add_argument('-p', '--pct_usage', type=float)
    parser.add_argument('-r', '--small_prop', type=float)
    parser.add_argument('-l', '--small_label', type=int)
    parser.add_argument('-m', '--aug_mode', type=str)
    parser.add_argument('-u', '--undersample', type=int)
    arg_dict = vars(parser.parse_args())
    arg_dict['gpu'] = 'cuda:'+str(arg_dict['gpu'])
    if arg_dict['undersample'] is not None:
        arg_dict['undersample'] = bool(arg_dict['undersample'])
    assert arg_dict['aug_mode'] is not None
    return arg_dict

def experiment(experiment_num):
    kwargs = arg_dict.copy()
    aug_mode = kwargs.pop('aug_mode')
    balance_seed, split_num = divmod(experiment_num, 10)
    logger = get_bert_logger(this_script_name, balance_seed, split_num=split_num, aug_mode=aug_mode)
    device = kwargs.pop('gpu')
    if kwargs.pop('only_original'):
        aug_mode = None
    else:
        from opt_params import sfu_params
        kwargs['geo'] = sfu_params[setting][aug_mode][key]
    kwargs['balance_seed'] = balance_seed
    kwargs['split_num'] = split_num
    agent = BertAgent(device, logger, data_name, 128, num_epochs, 
    				  lr, aug_mode, mode, batch_size, accumulation_steps,
    				  **kwargs)
    agent.run()


arg_dict = get_args()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 3
mode = 'test'
data_name = 'sfu'
batch_size = 32
accumulation_steps = 1
lr = 2e-5

if arg_dict['pct_usage'] is not None:
    assert arg_dict['small_prop'] is None
    setting = 'pct'
    key = arg_dict['pct_usage']
elif arg_dict['small_prop'] is not None:
    assert arg_dict['pct_usage'] is None
    setting = 'bal'
    key = arg_dict['small_prop']

experiment_num_list = list(range(arg_dict.pop('start_num'), arg_dict.pop('end_num')))
try:
	pool = mp.Pool(mp.cpu_count())
	pool.map(experiment, experiment_num_list)
finally:
	pool.close()
	pool.join()

# experiment(experiment_num_list[0])
