from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
import numpy as np
import pandas as pd
import os

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

# setting the spark context and app name 
conf = SparkConf().setMaster("local").setAppName("HousingJobSpark")
sc = SparkContext(conf=conf)

def load_data(file_path, file_name):
    full_path = os.path.join(file_path, file_name)
    return pd.read_csv(full_path)

def mapOceanProximity(proximity):

    if(proximity == "NEAR BAY"):
        return 1
    
    elif(proximity == "INLAND"):
        return 1
    
    elif(proximity =="<1H OCEAN"):
        return 0
    
    elif(proximity=="NEAR OCEAN"):
        return 1
    
    elif(proximity =="ISLAND"):
        return 1
    
    # else:
    #     return 0


def createLabeledPoint(fields):

    longitude = float(fields[0])
    latitude = float(fields[1])
    housing_median_age = float(fields[2])
    total_rooms = float(fields[3])
    total_bedrooms = int(fields[4])
    population = float(fields[4])
    households = float(fields[5])
    median_income = float(fields[6])
    median_house_value = float(fields[7])
    ocean_proximity = mapOceanProximity(fields[8])

    return LabeledPoint(ocean_proximity, np.array([longitude, latitude, housing_median_age, total_rooms,
                        total_bedrooms, population, households, median_income, median_house_value]))


DATA_PATH = os.path.join(os.getcwd(),'datasets', 'housing')
FILE_NAME = "housing.csv"
housing_data = load_data(DATA_PATH, FILE_NAME)

# print(housing_data)

raw_data_rdd = sc.textFile(os.path.join(DATA_PATH, FILE_NAME))
header  = raw_data_rdd.first()
raw_data_rdd = raw_data_rdd.filter(lambda x: x != header)

raw_data_rdd = raw_data_rdd.map(lambda x: x.split(","))

# training_data_rdd, test_data_rdd = raw_data_rdd.randomSplit(weights=[8.0,2.0], seed=1)  # raw_data_rdd.map(createLabeledPoint)

# training_data_rdd, test_data_rdd = training_data_rdd.map(createLabeledPoint), test_data_rdd.map(createLabeledPoint)

# model = NaiveBayes.train(training_data_rdd, 1.0)

# prediction_and_label = test_data_rdd.map(lambda p: (model.predict(p.features), p.label))

# accuracy = 1.0 * prediction_and_label.filter(lambda pl: pl[0] == pl[1]).count()/test_data_rdd.count()

# print("model accuracy {}".format(accuracy))







