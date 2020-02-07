import os

import numpy as np

from agents.rnn import RnnAgent
from agents.bert import BertAgent
from utils.logger import initialize_logger
from utils.parsing import get_config


# config = get_config()
this_script_name = os.path.basename(__file__).split('.')[0]
num_epochs = 4
aug_mode = None
batch_size = 16
for balance_seed in range(5):
	logger = initialize_logger(this_script_name, balance_seed)
	thing = BertAgent('foo', logger, 'sst', 25, num_epochs, 
					  aug_mode, 'dev', batch_size, 
					  small_label=None, small_prop=None,
					  balance_seed=balance_seed, undersample=False,
					  pct_usage=1)
# TRY UNDERSAMPLE=TRUE
thing.run()
exit()