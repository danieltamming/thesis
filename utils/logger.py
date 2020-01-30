import os
import time
import logging

def get_log_name(name):
	'''
	Logs are named by intergers, in order of construction. Gets index of 
	the next log. 
	'''
	folder_name = 'logs/'+name
	if not os.path.exists(folder_name): 
		os.mkdir(folder_name)
	log_name_list = [s for s in os.listdir(folder_name) 
					 if s.split('.')[0].isdigit()]
	if not log_name_list:
		n = 0
	else:
		n = max([int(s.split('.')[0]) for s in log_name_list]) + 1
	return os.path.join(folder_name, str(n)+'.log')


def initialize_logger(name):
	'''
	Creates new log file and returns logger that's independent of previous loggers
	'''
	log_file = get_log_name(name)
	handler = logging.FileHandler(log_file)
	logger = logging.getLogger(log_file)
	logger.setLevel(logging.INFO)
	logger.addHandler(handler)
	return logger

def print_and_log(logger, s):
	print(s)
	logger.info(s)