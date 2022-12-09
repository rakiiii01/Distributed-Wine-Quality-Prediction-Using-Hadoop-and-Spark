import random
import sys
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark_read = SparkSession.builder.appName("train").getOrCreate()
spark_read.sparkContext.setLogLevel("Error")

df_data= spark_read.read.format("csv").load(sys.argv[1], header=True, sep=";")


df_data = df_data.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")



df_data = df_data \
        .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
        .withColumn("volatile_acidity", col("volatile_acidity").cast(DoubleType())) \
        .withColumn("citric_acid", col("citric_acid").cast(DoubleType())) \
        .withColumn("residual_sugar", col("residual_sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free_sulfur_dioxide", col("free_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("total_sulfur_dioxide", col("total_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("label", col("label").cast(IntegerType()))


features_run = df_data.columns
features_run = features_run[:-1]


Assembler= VectorAssembler(inputCols=features_run, outputCol="features")
df_Assembler = Assembler.transform(df_data)
df_Assembler = df_Assembler.select(["features", "label"])
df1 = df_Assembler

maxIter=1000, 
blockSize=64, 
stepSize=0.04, 
solver='l-bfgs'
layers = [11, 8, 8, 8, 8, 10]
Model_create = MultilayerPerceptronClassifier(maxIter=maxIter, layers=layers, blockSize=blockSize, stepSize=stepSize, solver=solver)
Model_run = Model_create.fit(df1)
Model_run.write().overwrite().save(sys.argv[2])
print("Model_created")
