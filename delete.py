import os
import multiprocessing as mp
import time

from utils.logger import initialize_logger

def my_func(t):
	time.sleep(t)

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
start_time = time.time()
results = pool.map(my_func, list(range(1,6)))
print(time.time() - start_time)
pool.close()