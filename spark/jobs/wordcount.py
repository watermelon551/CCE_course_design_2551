import os
import time

from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("WordCount").getOrCreate()
sc = spark.sparkContext

input_path = os.getenv("WORDCOUNT_INPUT", "file:///opt/spark/work/sample.txt")

lines = sc.textFile(input_path)
word_counts = (
    lines.flatMap(lambda line: line.split())
    .map(lambda word: (word.lower(), 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda item: item[1], ascending=False)
)

print("Input path:", input_path)
print("Top 10 words:", word_counts.take(10))

sleep_seconds = int(os.getenv("WORDCOUNT_SLEEP_SECONDS", "120"))
print(f"Sleeping {sleep_seconds} seconds for screenshot capture.")
time.sleep(sleep_seconds)

spark.stop()
