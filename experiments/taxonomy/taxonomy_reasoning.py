# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import ltns.logictensornetworks_wrapper as ltnw
import tensorflow as tf
import numpy as np
import ltns.logictensornetworks as ltn
import itertools
import argparse

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

    folder_name = "reasoning_results/"

    entities = ["Cat", "Feline", "Mammal", "Agent", "Thing", "Dog", "Human",
                "Reptile", "Organization", "Company", "Animal", "Bank", "Snake", "Squirrel", "Dolphin", "Shark", "Bird",
                "Fish",
                "Lizard", "Crocodile", "BlueFish", "LilBird", "Eagle", "BaldEagle"]

    relationships = (
    ("Cat", "Feline"), ("Feline", "Mammal"), ("Mammal", "Animal"), ("Animal", "Agent"), ("Agent", "Thing"),
    ("Dog", "Mammal"), ("Human", "Mammal"), ("Organization", "Agent"), ("Company", "Organization"),
    ("Bank", "Company"), ("Snake", "Reptile"), ("Reptile", "Animal"),
    ("Dolphin", "Mammal"), ("Shark", "Fish"), ("Lizard", "Reptile"),
    ("Crocodile", "Reptile"), ("BlueFish", "Fish"),
    ("LilBird", "Bird"), ("Eagle", "Bird"), ("BaldEagle", "Bird"), ("Bird", "Animal"), ("Fish", "Animal"),
    ("Shark", "Fish"), ("Squirrel", "Mammal"))

    all_relationships = list(itertools.product(entities, repeat=2))

    ltnw.predicate("SubClass",embedding_size*2)

    for l in entities:
        ltnw.constant(l, min_value=[0.] * embedding_size, max_value=[1.] * embedding_size)

    for a,c in relationships:
        string = "SubClass(" + a + "," + c + ")"
        logging.info(string)
        ltnw.axiom(string)

    ltnw.variable("a",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
    ltnw.variable("b",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
    ltnw.variable("c",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))


    ltnw.axiom("forall a,b,c: (SubClass(a,b) & SubClass(b,c)) -> SubClass(a,c)")
    ltnw.axiom("forall a: ~SubClass(a,a)")
    ltnw.axiom("forall a,b: SubClass(a,b) -> ~SubClass(b,a)")

    ltnw.initialize_knowledgebase(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lr,decay=dc),
        formula_aggregator=lambda *x: tf.reduce_min(tf.concat(x, axis=0)))

    # Train the KB
    sat_level = ltnw.train(max_epochs=10000)

    print(sat_level)

    file_name = "tx_lr:" + str(lr) + "dc:" + str(dc) + "_em:" + str(embedding_size) + "_ly:" + str(ly) + "_bi:" + str(bi) + "_iter:" + str(iter_epoch) + "_univ:" + str(args.univ) + "_sat:" + np.array_str(sat_level)
    with open(folder_name + file_name, "w") as resutls_file:
        resutls_file.write(str(sat_level) + "\n")
        logging.info("Inference Over Taxonomy")
        for a,b in all_relationships:
            resutls_file.write(a + "," + b + "," + np.array_str(ltnw.ask("SubClass("+ a +"," + b +")").squeeze()) + "\n")




