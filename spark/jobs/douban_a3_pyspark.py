import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, explode, round as spark_round, split, trim
from pyspark.sql.types import DoubleType, IntegerType, LongType


def print_rows(rows):
    headers = ["genre", "movie_count", "avg_rating", "avg_rating_count"]
    widths = [12, 12, 12, 18]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        values = [
            str(row["genre"]),
            str(row["movie_count"]),
            str(row["avg_rating"]),
            str(row["avg_rating_count"]),
        ]
        print(" | ".join(value.ljust(width)[:width] for value, width in zip(values, widths)))


spark = SparkSession.builder.appName("DoubanA3GenreGroupBy").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

input_path = os.getenv("DOUBAN_INPUT", "file:///opt/spark/work/douban_movies.csv")
executor_label = os.getenv("A3_EXECUTORS", "unknown")

print("========== A-3 PySpark Genre GROUP BY ==========")
print("Execution mode: PySpark")
print("Executor instances:", executor_label)
print("Input path:", input_path)

start = time.perf_counter()

raw = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("quote", '"')
    .option("escape", '"')
    .csv(input_path)
)

movies = (
    raw.dropna(subset=["genres", "rating_score", "year"])
    .withColumn("year", col("year").cast(IntegerType()))
    .withColumn("rating_score", col("rating_score").cast(DoubleType()))
    .withColumn("rating_count", col("rating_count").cast(LongType()))
)

genre_movies = (
    movies.withColumn("genre", explode(split(col("genres"), "/")))
    .withColumn("genre", trim(col("genre")))
    .filter(col("genre") != "")
)

result = (
    genre_movies.groupBy("genre")
    .agg(
        count("*").alias("movie_count"),
        spark_round(avg("rating_score"), 3).alias("avg_rating"),
        spark_round(avg("rating_count"), 1).alias("avg_rating_count"),
    )
    .filter(col("movie_count") >= 100)
    .orderBy(col("movie_count").desc())
    .limit(15)
)

rows = result.collect()
elapsed = time.perf_counter() - start

print_rows(rows)
print(f"PYSPARK_EXECUTOR_INSTANCES: {executor_label}")
print(f"PYSPARK_ELAPSED_SECONDS: {elapsed:.6f}")

spark.stop()
