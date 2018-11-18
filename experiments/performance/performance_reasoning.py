# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import ltns.logictensornetworks_wrapper as ltnw
import tensorflow as tf
import numpy as np
import ltns.logictensornetworks as ltn
import matplotlib.pyplot as plt
from ltns.logictensornetworks import Not,And,Implies,Forall,Exists,Equiv
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
import argparse
import itertools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-const', '--const',
                        help="", type=int)
    parser.add_argument('-pred', '--pred',
                        help="", type=int)
    parser.add_argument('-card', '--card',
                        help="", type=int)

    args = parser.parse_args()


    pred = args.pred
    const = args.const
    card = args.card

    ltn.LAYERS = 4
    ltn.BIAS_factor = -1e-8
    ltn.set_universal_aggreg("mean") # The truth value of forall x p(x) is
                                     # interpretable as the percentage of
                                     # element in the range of x that satisties p

    embedding_size = 4 # embedding space dimensionality

    pred_name = "pred"
    const_name = "const"

    predicates = list(map(lambda x : pred_name + str(x), range(1, 100)))
    constants = list(map(lambda x : const_name + str(x), range(1, 100)))

    for l in constants[:const]:
        print(l)
        ltnw.constant(l, min_value=[0.] * embedding_size, max_value=[1.] * embedding_size)


    if card == 1:
        ltnw.variable("x", tf.concat(list(ltnw.CONSTANTS.values()), axis=0))
        for k in predicates[:pred]:
            ltnw.predicate(k, embedding_size)
            string = "forall x : " + k + "(x)"
            print(string)
            ltnw.axiom(string)
    elif card == 2:
        ltnw.variable("x", tf.concat(list(ltnw.CONSTANTS.values()), axis=0))
        ltnw.variable("y", tf.concat(list(ltnw.CONSTANTS.values()), axis=0))
        for k in predicates[:pred]:
            ltnw.predicate(k, embedding_size * 2)
            string ="forall x,y : " + k + "(x,y)"
            print(string)
            ltnw.axiom(string)
    else:
        ltnw.variable("x", tf.concat(list(ltnw.CONSTANTS.values()), axis=0))
        ltnw.variable("y", tf.concat(list(ltnw.CONSTANTS.values()), axis=0))
        ltnw.variable("z", tf.concat(list(ltnw.CONSTANTS.values()), axis=0))
        for k in predicates[:pred]:
            ltnw.predicate(k, embedding_size * 3)
            string ="forall x,y,z : " + k + "(x,y,z)"
            print(string)
            ltnw.axiom(string)


    ltnw.initialize_knowledgebase(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9),
        formula_aggregator=lambda *x: tf.reduce_min(tf.concat(x, axis=0)))


    # Train the KB
    import time
    with open("performance_results/performance_1", "a") as timereason:
        start_time = time.time()
        sat_level = ltnw.train(max_epochs=5000, track_sat_levels=5000)
        end_time = time.time()
        timereason.write(str(pred) + "," +str(const) + "," + str(end_time - start_time) + "\n")
    print(pred, const, end_time - start_time)

