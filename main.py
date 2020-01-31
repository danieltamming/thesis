import os

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_config


# config = get_config()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
batch_size = 32
balance_seed = 0
undersample = True
logger = initialize_logger(this_script_name, balance_seed)
thing = BertAgent('foo', logger, 'sst', 25, num_epochs, None, 'dev', batch_size, 
				 small_label=0, small_prop=0.5,
				 balance_seed=balance_seed, undersample=undersample)
# TRY UNDERSAMPLE=TRUE
thing.run()
exit()