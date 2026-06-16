from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("DoubanAnalysisPlaceholder").getOrCreate()

print("PySpark image is ready. Use this file for A-1 and A-2 analysis later.")

spark.stop()
