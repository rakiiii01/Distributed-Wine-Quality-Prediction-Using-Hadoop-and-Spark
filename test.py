import random
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark_session = SparkSession.builder.appName("test").getOrCreate()
spark_session.sparkContext.setLogLevel("Error")


df_session = spark_session.read.format("csv").load(sys.argv[1], header=True, sep=";")


df_session = df_session.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")

df_session = df_session \
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


features_run = df_session.columns
features_run = features_run[:-1]
Assesmbler = VectorAssembler(inputCols=features_run, outputCol="features")
df_Assesmbler = Assesmbler.transform(df_session)
df_Assembler = df_Assesmbler.select(["features", "label"])
df = df_Assembler
Model_run = MultilayerPerceptronClassificationModel.load(sys.argv[2])
result = Model_run.transform(df)
class_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc_evaluate = class_evaluator.evaluate(result)
print("############################")
print("Accuracy : %g " % acc_evaluate )
print("############################")
evaluate_res = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_res = evaluate_res.evaluate(result)
print("############################")
print("F1-score : %g " % evaluator_res)
print("############################")

