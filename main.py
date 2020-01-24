# from agents.bilstm import BiLSTMAgent
from agents.rnn import RnnAgent
from agents.bert import BertAgent

from utils.logger import initialize_logger
from utils.parsing import get_config


# config = get_config()
initialize_logger()
thing = RnnAgent('foo', 'sst', 25, 50, None, 'dev', 128, 0, 0.5)
thing.run()

for geo in [0.1, 0.3, 0.5, 0.7, 0.9]:
	initialize_logger()
	thing = RnnAgent('foo', 'sst', 25, 50, 'synonym', 'dev', 128, 0, 0.5, geo=geo)
	thing.run()


# thing = BertAgent('foo', 'sst', 25, 4, 'synonym', 'dev', 32)