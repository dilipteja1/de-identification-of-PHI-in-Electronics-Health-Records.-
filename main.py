'''
Author : dilip teja
'''
import datetime
import json
import os
import re
import pandas as pd
import random
import xml.etree.ElementTree as ET # to read the xml files
from nltk import sent_tokenize

# TRAINING_SET1 = "dataset/2014_training-PHI-Gold-Set1/training-PHI-Gold-Set1"
TRAINING_SET1 = "dataset/test"

# TRAINING_SET2 = "dataset/training-PHI-Gold-Set2/training-PHI-Gold-Set2"

TESTING_SET = "dataset/testing-PHI-Gold-fixed/testing-PHI-Gold-fixed"

def load_data():

    train_filelist = []
    test_filelist = []
    for file in os.listdir(TRAINING_SET1):
        if ".xml" in file:
            train_filelist.append(os.path.join(TRAINING_SET1, file))
    # for file in os.listdir(TRAINING_SET2):
    #     if ".xml" in file:
    #         train_filelist.append(os.path.join(TRAINING_SET2, file))
    for file in os.listdir(TESTING_SET):
        if ".xml" in file:
            test_filelist.append(os.path.join(TESTING_SET, file))

    print(f"loading the training data : {len(train_filelist)}")
    print(f"loading the testing data: {len(test_filelist)}")
    return train_filelist, test_filelist


from model import BertBaseLine

if __name__ == "__main__":
    # bert baseline
    bert_model_class = BertBaseLine()
    training_files, testing_files = load_data()
    bert_model_class.prepare_dataset(training_files, testing_files)
    # bert_model_class.train()
