from utils import evaluation_utilities
import pprint

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


import os
from os import listdir
from os.path import isfile, join


closure = os.getcwd() + "/../gold_standard/closed_an"
file_dir = os.getcwd()+ "/reasoning_results/"

onlyfiles = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]

tuples_file_scores = []
onlyfiles = filter(lambda x : "pa" not in x, onlyfiles)

for item in onlyfiles:
    scores = (evaluation_utilities.compute_all_values(closure, file_dir + item, parents, False))
    tuples_file_scores.append((item, scores))

pprint.pprint(sorted(tuples_file_scores, key=lambda x: x[1]["f1"], reverse=True)[0])