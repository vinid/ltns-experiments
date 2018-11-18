import itertools
import os

learn_rates = [0.01]
decay_param= [0.9]
layers = [10, 20]
embedding_sizes = [10,20]
biases_index = [1]
universal_aggregator = ["mean", "hmean", "min"]

data_pairs = list(itertools.product(learn_rates, decay_param, layers, embedding_sizes, biases_index, universal_aggregator))

max_iter = len(data_pairs)
current = 0

for lr, dc, ly, em, bi, uv in data_pairs:
    cwd = os.getcwd()
    for iter in range(1, 5):
        print current, "on", max_iter*4
        current = current + 1
        os.system("python " + cwd + "/ancestors/ancestor_reasoning_extended.py -lr " + str(lr) + " -dc " + str(dc) + " -ly  " \
                  + str(ly) + " -em " + str(em) + " -bi " + str(bi) + " -iter " + str(iter) + " -univ " + str(uv))