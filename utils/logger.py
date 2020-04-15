import os
import time
import logging

def get_log_name(name, seed, other=None):
	'''
	Logs are named by intergers, in order of construction. Gets index of 
	the next log. 
	'''
	folder_name = 'logs/'+name
	os.makedirs(folder_name, exist_ok=True)
	log_name_list = [s for s in os.listdir(folder_name) 
					 if 'seed_{}'.format(seed) in s]
	if not log_name_list:
		num = 0
	else:
		num = max([int(s.split('.')[0][-1]) for s in log_name_list]) + 1
	if other is None:
		filename = 'seed_{}_num_{}.log'.format(seed, num)
	else:
		filename = 'seed_{}_other_{}_num_{}.log'.format(seed, other, num)
	return os.path.join(folder_name, filename)

def logger_init(log_file):
	handler = logging.FileHandler(log_file)
	logger = logging.getLogger(log_file)
	logger.setLevel(logging.INFO)
	logger.addHandler(handler)
	return logger

def initialize_logger(name, seed, other=None):
	'''
	Creates new log file and returns logger that's independent of previous loggers
	'''
	log_file = get_log_name(name, seed, other)
	return logger_init(log_file)

def get_bert_logger(name, seed):
	folder_name = 'logs/' + name
	os.makedirs(folder_name, exist_ok=True)
	filename = 'seed_{}.log'.format(seed)
	log_file = os.path.join(folder_name, filename)
	return logger_init(log_file)

def print_and_log(logger, s):
	print(s)
	logger.info(s)