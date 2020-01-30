import os
import multiprocessing as mp
import time

from utils.logger import initialize_logger

logger = initialize_logger(os.path.basename(__file__).split('.')[0])
logger.info('Hello')
time.sleep(3)
logger = initialize_logger(os.path.basename(__file__).split('.')[0])
logger.info('Goodbye')
exit()
print(os.path.basename(__file__).split('.')[0])
time.sleep(3)
initialize_logger('temp')
exit()

def my_func(t, k):
	time.sleep(t)
	return (t, k)

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
start_time = time.time()
results = pool.starmap(my_func, [(4, 1), (3, 2), (2, 3), (1, 4), (5, 5)])
print(time.time() - start_time)
pool.close()
print(results)