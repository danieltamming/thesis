# from agents.bilstm import BiLSTMAgent
from agents.rnn import RnnAgent
from agents.bert import BertAgent

from utils.logger import initialize_logger
from utils.parsing import get_config


# config = get_config()
initialize_logger()
thing = BertAgent('foo', 'sst', 25, 50, 'synonym', 'dev', 128, 
				 small_label=0, small_prop=0.5,
				 balance_seed=1, undersample=True, geo=0.5)
thing.run()
# exit()

for balance_seed in range(5):
	initialize_logger()
	thing = RnnAgent('foo', 'sst', 25, 50, None, 'dev', 128, 
					 small_label=0, small_prop=0.5, 
					 balance_seed=balance_seed)
	thing.run()
	for geo in [0.5, 0.7, 0.9]:
		initialize_logger()
		thing = RnnAgent('foo', 'sst', 25, 100, 'synonym', 'dev', 128, 
						 small_label=0, small_prop=0.5, 
						 balance_seed=balance_seed, geo=geo)
		thing.run()