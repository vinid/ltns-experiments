from utils import evaluation_utilities
import pprint

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


import os
from os import listdir
from os.path import isfile, join


closure = os.getcwd() + "/../gold_standard/closed_tx"
file_dir = os.getcwd()+ "/reasoning_results/"

onlyfiles = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]

tuples_file_scores = []

for item in onlyfiles:
    scores = (evaluation_utilities.compute_all_values(closure, file_dir + item, relationships, True))
    tuples_file_scores.append((item, scores))

pprint.pprint(sorted(tuples_file_scores, key=lambda x: x[1]["f1"], reverse=True)[0])