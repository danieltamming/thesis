import os
import time
import logging

def initialize_logger(name=None):
	if not os.path.exists('logs/'): 
		os.mkdir('logs/')
	if name is None:
		logname = time.strftime('logs/%Y:%m:%d-%H:%M:%S.log')
	else:
		logname = 'logs/'+name+'.log'
	logging.basicConfig(filename=logname, level=logging.DEBUG)
	if name is not None:
		logging.info('------------------------RUN START------------------------')
	# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def print_and_log(logger, s):
	print(s)
	logger.info(s)