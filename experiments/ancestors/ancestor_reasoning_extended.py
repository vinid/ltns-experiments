# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import ltns.logictensornetworks_wrapper as ltnw
import tensorflow as tf
import numpy as np
import ltns.logictensornetworks as ltn
import argparse
import itertools

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr',
                        help="", type=float)
    parser.add_argument('-dc', '--dc',
                        help="", type=float)
    parser.add_argument('-ly', '--ly',
                        help="", type=int)
    parser.add_argument('-em', '--em',
                        help="", type=int)
    parser.add_argument('-bi', '--bi',
                        help="", type=int)
    parser.add_argument('-iter', '--iter',
                        help="", type=int)
    parser.add_argument('-univ', '--univ',
                        help="", type=str)

    args = parser.parse_args()
    biases = [-1e-8, -1e-5, -1e-1]

    lr = args.lr
    dc = args.dc
    ly = args.ly
    embedding_size = args.em

    bi = biases[args.bi]
    iter_epoch = args.iter

    ltn.LAYERS = ly
    ltn.BIAS_factor = bi
    ltn.set_universal_aggreg(args.univ)

    folder_name = "ancestors/reasoning_results_over_extended_axioms/"

    entities = ["sue", "diana", "john", "edna", "paul", "francis", "john2",
                "john3", "john4", "joe", "jennifer", "juliet", "janice",
                "joey", "tom", "bonnie", "katie"]

    parents = [
        ("sue", "diana"),
        ("john", "diana"),
        ("sue", "bonnie"),
        ("john", "bonnie"),
        ("sue", "tom"),
        ("john", "tom"),
        ("diana", "katie"),
        ("paul", "katie"),
        ("edna", "sue"),
        ("john2", "sue"),
        ("edna", "john3"),
        ("john2", "john3"),
        ("francis", "john"),
        ("john4", "john"),
        ("francis", "janice"),
        ("john4", "janice"),
        ("janice", "jennifer"),
        ("joe", "jennifer"),
        ("janice", "juliet"),
        ("joe", "juliet"),
        ("janice", "joey"),
        ("joe", "joey")]

    ltnw.predicate("ancestor",embedding_size*2)
    ltnw.predicate("parent",embedding_size*2)

    for l in entities:
        ltnw.constant(l, min_value=[0.] * embedding_size, max_value=[1.] * embedding_size)

    for a,c in parents:
        string = "parent(" + a + "," + c + ")"
        logging.info(string)
        ltnw.axiom(string)

    ltnw.variable("a",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
    ltnw.variable("b",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
    ltnw.variable("c",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))

    ltnw.axiom("forall a,b: parent(a,b) -> ancestor(a,b)")
    ltnw.axiom("forall a,b,c: (ancestor(a,b) &  parent(b,c)) -> ancestor(a,c)")
    ltnw.axiom("forall a,b,c: (parent(a,b) &  parent(b,c)) -> ancestor(a,c)")
    ltnw.axiom("forall a,b,c: (ancestor(a,b) & ancestor(b,c)) -> ancestor(a,c)")
    ltnw.axiom("forall a: ~parent(a,a)")
    ltnw.axiom("forall a: ~ancestor(a,a)")
    ltnw.axiom("forall a,b: parent(a,b) -> ~parent(b,a)")
    ltnw.axiom("forall a,b: ancestor(a,b) -> ~ancestor(b,a)")


    ltnw.initialize_knowledgebase(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9),
        formula_aggregator=lambda *x: tf.reduce_min(tf.concat(x, axis=0)))


    # Train the KB
    sat_level = ltnw.train(max_epochs=10000)

    all_relationships = list(itertools.product(entities, repeat=2))

    file_name_an = "an_lr:" + str(lr) + "dc:" + str(dc) + "_em:" + str(embedding_size) + "_ly:" + str(ly) + "_bi:" + str(bi) + "_iter:" + str(iter_epoch) + "_univ:" + str(args.univ) + "_sat:" + np.array_str(sat_level)
    file_name_pa = "pa_lr:" + str(lr) + "dc:" + str(dc) + "_em:" + str(embedding_size) + "_ly:" + str(ly) + "_bi:" + str(bi) + "_iter:" + str(iter_epoch) + "_univ:" + str(args.univ) + "_sat:" + np.array_str(sat_level)

    with open(folder_name + file_name_an, "w") as resutls_file:
        resutls_file.write(str(sat_level) + "\n")
        logging.info("Inferencing Ancestors")
        for a,b in all_relationships:
            resutls_file.write(a + "," + b + "," + np.array_str(ltnw.ask("ancestor("+ a +"," + b +")").squeeze()) + "\n")

    with open(folder_name + file_name_pa, "w") as resutls_file:
        resutls_file.write(str(sat_level) + "\n")
        logging.info("Inferencing Parents")
        for a,b in all_relationships:
            resutls_file.write(a + "," + b + "," + np.array_str(ltnw.ask("parent("+ a +"," + b +")").squeeze()) + "\n")




