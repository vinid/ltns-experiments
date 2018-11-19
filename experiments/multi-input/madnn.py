import pandas as pd
import os
import numpy as np
from keras.models import Model
from keras import layers, regularizers
from keras.layers import Input
import logging
import ltns.logictensornetworks as ltn

logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import ltns.logictensornetworks_wrapper as ltnw
import tensorflow as tf


ltn.LAYERS = 10
embedding_size = 10

ltn.BIAS_factor = -1e-5
ltn.set_universal_aggreg("hmean")

closed_pa = pd.read_csv(os.getcwd() + "/../gold_standard/closed_pa", names=["first", "second", "type"])
closed_an = pd.read_csv(os.getcwd() + "/../gold_standard/closed_an", names=["first", "second", "type"])

#training_pa = closed_pa.sample(100)
#training_an = closed_an.sample(100)

#test_pa = (pd.concat([training_pa,closed_pa]).drop_duplicates(keep=False))
#test_an = (pd.concat([training_an,closed_an]).drop_duplicates(keep=False))

training_pa = pd.read_csv("training_testing/training_pa")
training_an = pd.read_csv("training_testing/training_an")
test_pa = pd.read_csv("training_testing/test_pa")
test_an = pd.read_csv("training_testing/test_an")

#training_pa.to_csv("training_testing/training_pa", index=False)
#training_an.to_csv("training_testing/training_an", index=False)
#test_pa.to_csv("training_testing/test_pa", index=False)
#test_an.to_csv("training_testing/test_an", index=False)

entities = ["sue", "diana", "john", "edna", "paul", "francis", "john2",
            "john3", "john4", "joe", "jennifer", "juliet", "janice", "joey", "tom", "bonnie", "katie"]

def ltnsnet():
    ltnw.predicate("ancestor",embedding_size*2)
    ltnw.predicate("parent",embedding_size*2)

    for l in entities:
        ltnw.constant(l, min_value=[0.] * embedding_size, max_value=[1.] * embedding_size)

    for index, row in training_pa.iterrows():
        if row["type"] == 1:
            string = "parent(" + row["first"] + "," + row["second"] + ")"
        else:
            string = "~parent(" + row["first"] + "," + row["second"] + ")"
        ltnw.axiom(string)

    for index, row in training_an.iterrows():
        if row["type"] == 1:
            string = "ancestor(" + row["first"] + "," + row["second"] + ")"
        else:
            string = "~ancestor(" + row["first"] + "," + row["second"] + ")"
        ltnw.axiom(string)


    ltnw.variable("a",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
    ltnw.variable("b",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
    ltnw.variable("c",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))

    ltnw.axiom("forall a,b: parent(a,b) -> ancestor(a,b)")
    ltnw.axiom("forall a,b,c: (ancestor(a,b) &  parent(b,c)) -> ancestor(a,c)")
    ltnw.axiom("forall a: ~parent(a,a)")
    ltnw.axiom("forall a: ~ancestor(a,a)")
    ltnw.axiom("forall a,b: parent(a,b) -> ~parent(b,a)")
    ltnw.axiom("forall a,b: ancestor(a,b) -> ~ancestor(b,a)")

    ltnw.initialize_knowledgebase(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9),
        formula_aggregator=lambda *x: tf.reduce_min(tf.concat(x, axis=0)))

    # Train the KB
    sat_level = ltnw.train(max_epochs=10000)

    print(sat_level)

    file_name_an = "an_prediction"
    file_name_pa = "pa_prediction"

    with open(file_name_an, "w") as resutls_file:
        resutls_file.write(str(sat_level) + "\n")
        print("inferencing an")
        for index,row in test_an.iterrows():
            resutls_file.write(row["first"] + "," + row["second"] + "," + np.array_str(ltnw.ask("ancestor("+ row["first"] +"," + row["second"] +")").squeeze()) + "\n")

    with open(file_name_pa, "w") as resutls_file:
        resutls_file.write(str(sat_level) + "\n")
        print("inferencing pa")
        for index, row in test_pa.iterrows():
            resutls_file.write(row["first"] + "," + row["second"] + "," + np.array_str(ltnw.ask("parent("+ row["first"] +"," + row["second"] +")").squeeze()) + "\n")


word2int = {}
int2word = {}


def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

vocab_size = len(entities)  # gives the total number of unique words

for i, word in enumerate(entities):
    word2int[word] = i
    int2word[i] = word

parent_vec = [0, 1]
ancestor_vec = [1, 0]

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper



def madnn():

    entity_one_train = []  # input word
    entity_two_train = []  # input word
    property_train = []
    y_train = []  # output word

    for index, row in training_pa.iterrows():
        first = to_one_hot(word2int[row["first"]], vocab_size)
        second = to_one_hot(word2int[row["second"]], vocab_size)
        entity_one_train.append(first)
        entity_two_train.append(second)
        property_train.append(parent_vec)
        y_train.append([row["type"]])

    for index, row in training_an.iterrows():
        first = to_one_hot(word2int[row["first"]], vocab_size)
        second = to_one_hot(word2int[row["second"]], vocab_size)
        entity_one_train.append(first)
        entity_two_train.append(second)
        property_train.append(parent_vec)
        y_train.append(np.array([row["type"]]))

    # convert them to numpy arrays
    entity_one_train = np.asarray(entity_one_train)
    entity_two_train = np.asarray(entity_two_train)
    property_train = np.asarray(property_train)
    y_train = np.asarray(y_train)

    test_entity_one_train = []  # input word
    test_entity_two_train = []  # input word
    test_property_train = []
    test_y_train = []  # output word

    for index, row in test_an.iterrows():
        first = to_one_hot(word2int[row["first"]], vocab_size)
        second = to_one_hot(word2int[row["second"]], vocab_size)
        test_entity_one_train.append(first)
        test_entity_two_train.append(second)
        test_property_train.append(parent_vec)
        test_y_train.append(np.array([row["type"]]))

    # convert them to numpy arrays
    test_entity_one_train = np.asarray(test_entity_one_train)
    test_entity_two_train = np.asarray(test_entity_two_train)
    test_property_train = np.asarray(test_property_train)
    test_y_train = np.asarray(test_y_train)


    entity_vocabulary_size = len(entities)
    prop_vocabulary_size = 2

    entity_one_input = Input(shape=(entity_vocabulary_size,))
    entity_one_first_step = layers.Dense(5,kernel_regularizer=regularizers.l2(0.01))(entity_one_input)

    entity_two_input = Input(shape=(entity_vocabulary_size,))
    entity_two_first_step = layers.Dense(5 ,kernel_regularizer=regularizers.l2(0.01))(entity_two_input)

    property_input = Input(shape=(prop_vocabulary_size,))
    encoded_property = layers.Dense(2, kernel_regularizer=regularizers.l2(0.01))(property_input)

    concatenated = layers.concatenate([entity_one_first_step, encoded_property, entity_two_first_step], axis=-1)

    belta = layers.Dense(3, kernel_regularizer=regularizers.l2(0.01))(concatenated)
    answer = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(belta)

    model = Model([entity_one_input, entity_two_input,property_input], answer)
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy", precision, recall])

    model.fit([entity_one_train,entity_two_train, property_train], y_train, epochs=1000, batch_size=20, validation_split=0.30)

    loss, accuracy,precision,recall = (model.evaluate([test_entity_one_train,test_entity_two_train, test_property_train], test_y_train))

    return accuracy,precision,recall
#ltnsnet()
print(madnn())
