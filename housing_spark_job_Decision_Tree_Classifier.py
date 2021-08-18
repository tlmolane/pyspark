from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext , SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
import pandas as pd
import os
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, FloatType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import when, lit, col


spark = SparkSession.builder.appName("Job1").master("local").getOrCreate()
sc = SparkContext

DATA_PATH = os.path.join("/home/zeefu/Desktop")
FILE_NAME = "housing.csv"
FULL_PATH = os.path.join(DATA_PATH, FILE_NAME)

# housingfileSchema = StructType([
#     StructField('longitude', FloatType(), True),
#     StructField('latitude', FloatType(), True),
#     StructField('housing_median_age', FloatType(), True),
#     StructField('total_rooms', FloatType(), True),
#     StructField('total_bedrooms', FloatType(), True),
#     StructField('population', FloatType(), True),
#     StructField('households', FloatType(), True),
#     StructField('median_income', FloatType(), True),
#     StructField('median_house_value', FloatType(), True),
#     StructField('ocean_proximity', StringType(), True)
# ])

# df = spark.read.option("delimiter", ",").schema(housingfileSchema).csv(FULL_PATH, header=True)

df = spark.read.csv(FULL_PATH, header = True, inferSchema = True)

#### Some Feature Engineering 

# replace null values with median value for all coloumns // Note: careful of categorical features such as ocean_proximity...
# in this case we do not have missing values in this column so no value is going to be replaced!

def replace(df, column, value, quantile = None):
    if quantile != None:
        quant = df.approxQuantile(column, [0.5], 0)
        return when(column !=value, column).otherwise(quant)
    else:
        return when(column !=value, column)

df_clean = df
df_schema_names = df.schema.names
outcome_label = "ocean_proximity"
df_schema_names.remove(outcome_label)

# for column_name in df_schema_names:
#     df_clean.withColumn(column_name, replace(df_clean, col(column_name), lit(None)))

for column_name in df_schema_names:
    df_clean = df_clean.filter(F.col(column_name).isNotNull())

# df_clean_rdd = df_clean.rdd

# # Split Data Into Training Data & Test Data 
# training_data_rdd, test_data_rdd = df_clean_rdd.randomSplit(weights=[8.0,2.0], seed=1)  # raw_data_rdd.map(createLabeledPoint)

data= df_clean.na.replace(['NEAR BAY', 'INLAND','<1H OCEAN', 'NEAR OCEAN', 'ISLAND'], ['1','0','1','1','0'], 'ocean_proximity')

data = data.withColumn("ocean_proximity", data.ocean_proximity.cast('int'))

train_features = [feature for feature in df_schema_names]

print("numeric features: {}".format(train_features))

assembler = VectorAssembler(inputCols=train_features, outputCol='features')

transformed_data = assembler.transform(data)
(training_data, test_data) = transformed_data.randomSplit([0.8,0.2])

rf = RandomForestClassifier(labelCol='ocean_proximity', 
                            featuresCol='features',
                            maxDepth=5)


evaluator = BinaryClassificationEvaluator()
model = rf.fit(training_data)

rf_predictions = model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'ocean_proximity', metricName = 'accuracy')
print('Random Forest classifier Accuracy:', multi_evaluator.evaluate(rf_predictions))
