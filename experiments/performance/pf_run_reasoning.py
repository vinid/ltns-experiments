import itertools
import os
import time
preds = [4, 8, 12, 20, 30]
consts= [4, 8, 12, 20, 30]
import numpy as np
data_pairs = list(itertools.product(preds, consts))

max_iter = len(data_pairs)
current = 0
values = []
for pred, const in data_pairs:
    cwd = os.getcwd()
    pair_time = []
    for iter in range(1, 4):
        print current, "on", max_iter*3
        current = current + 1
        start_time = time.time()
        os.system("python " + cwd + "/performance_reasoning.py -card " \
                  + str(1) + " -pred " + str(pred) + " -const  " + str(const))
        end_time = time.time()


print values