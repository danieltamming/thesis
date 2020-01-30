import os
import time
import logging

def get_log_name(name, seed):
	'''
	Logs are named by intergers, in order of construction. Gets index of 
	the next log. 
	'''
	folder_name = 'logs/'+name
	if not os.path.exists(folder_name): 
		os.mkdir(folder_name)
	log_name_list = [s for s in os.listdir(folder_name) 
					 if 'seed_{}'.format(seed) in s]
	if not log_name_list:
		num = 0
	else:
		num = max([int(s.split('.')[0][-1]) for s in log_name_list]) + 1
	filename = 'seed_{}_num_{}.log'.format(seed, num)
	return os.path.join(folder_name, filename)


def initialize_logger(name, seed):
	'''
	Creates new log file and returns logger that's independent of previous loggers
	'''
	log_file = get_log_name(name, seed)
	handler = logging.FileHandler(log_file)
	logger = logging.getLogger(log_file)
	logger.setLevel(logging.INFO)
	logger.addHandler(handler)
	return logger

def print_and_log(logger, s):
	print(s)
	logger.info(s)