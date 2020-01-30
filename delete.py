import os
import multiprocessing as mp
import time

from utils.logger import initialize_logger

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