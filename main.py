import numpy as np

# from agents.bilstm import BiLSTMAgent
from agents.rnn import RnnAgent
from agents.bert import BertAgent

from utils.logger import initialize_logger
from utils.parsing import get_config


# config = get_config()
initialize_logger()
thing = RnnAgent('foo', 'sst', 25, 50, None, 'dev', 128, 
				 small_label=0, small_prop=0.5,
				 balance_seed=1, undersample=True)
# TRY UNDERSAMPLE=TRUE
thing.run()
# exit()


num_epochs = 100
for balance_seed in range(5):
	initialize_logger()
	thing = RnnAgent('foo', 'sst', 25, num_epochs, None, 'dev', 128, 
					 small_label=0, small_prop=0.5, 
					 balance_seed=balance_seed)
	thing.run()
	for geo in range(0.1, 1.0, 0.1):
		initialize_logger()
		thing = RnnAgent('foo', 'sst', 25, num_epochs, 'trans', 'dev', 128, 
						 small_label=0, small_prop=0.5, 
						 balance_seed=balance_seed, geo=geo)
		thing.run()

'''
num_epochs = 100
for balance_seed in range(5):
	# initialize_logger()
	# thing = RnnAgent('foo', 'sst', 25, num_epochs, None, 'dev', 128, 
	# 				 small_label=0, small_prop=0.5, 
	# 				 balance_seed=balance_seed)
	# thing.run()
	for geo in [0.3, 0.4, 0.6]:
		initialize_logger()
		thing = RnnAgent('foo', 'sst', 25, num_epochs, 'synonym', 'dev', 128, 
						 small_label=0, small_prop=0.5, 
						 balance_seed=balance_seed, geo=geo)
		thing.run()
'''