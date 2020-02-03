import os

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_config


# config = get_config()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 100
aug_mode = 'synonym'
batch_size = 32
balance_seed = 0
logger = initialize_logger(this_script_name, balance_seed)
thing = RnnAgent('foo', logger, 'sst', 25, num_epochs, 
				  aug_mode, 'dev', batch_size, 
				  small_label=None, small_prop=None,
				  balance_seed=balance_seed, undersample=False,
				  pct_usage=1, geo=0.5)
# TRY UNDERSAMPLE=TRUE
thing.run()
exit()