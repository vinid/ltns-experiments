from utils import evaluation_utilities
import pprint
import os

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



closure = os.getcwd() + "/training_testing/test_an"
file_dir = os.getcwd()+ "/an_prediction"


scores = (evaluation_utilities.compute_all_values_ml(closure, file_dir))
pprint.pprint(scores)
