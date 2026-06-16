import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, explode, row_number, round as spark_round, split, trim
from pyspark.sql.types import DoubleType, IntegerType, LongType
from pyspark.sql.window import Window


spark = SparkSession.builder.appName("DoubanMovieSqlAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

input_path = os.getenv("DOUBAN_INPUT", "file:///opt/spark/work/douban_movies.csv")

print("========== A-2 Douban Movie Spark SQL Analysis ==========")
print("Input path:", input_path)

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
    .fillna({"directors": "未知导演", "countries": "未知地区", "summary": "暂无简介"})
    .withColumn("year", col("year").cast(IntegerType()))
    .withColumn("rating_score", col("rating_score").cast(DoubleType()))
    .withColumn("rating_count", col("rating_count").cast(LongType()))
    .withColumn("collect_count", col("collect_count").cast(LongType()))
)

movies.createOrReplaceTempView("movies")
print("Cleaned rows for A-2:", movies.count())

genre_movies = (
    movies.withColumn("genre", explode(split(col("genres"), "/")))
    .withColumn("genre", trim(col("genre")))
    .filter(col("genre") != "")
)
genre_movies.createOrReplaceTempView("genre_movies")

print("\n========== Query 1: GROUP BY Genre Aggregation ==========")
q1 = spark.sql(
    """
    SELECT
      genre,
      COUNT(*) AS movie_count,
      ROUND(AVG(rating_score), 3) AS avg_rating,
      ROUND(AVG(rating_count), 1) AS avg_rating_count
    FROM genre_movies
    GROUP BY genre
    HAVING movie_count >= 100
    ORDER BY movie_count DESC
    LIMIT 15
    """
)
q1.show(15, truncate=30)

print("\n========== Query 2: ORDER BY Top-N Movies ==========")
q2 = spark.sql(
    """
    SELECT
      title,
      original_title,
      year,
      rating_score,
      rating_count,
      collect_count
    FROM movies
    ORDER BY rating_score DESC, rating_count DESC
    LIMIT 10
    """
)
q2.show(10, truncate=28)

print("\n========== Query 3: Time Trend By Year ==========")
q3 = spark.sql(
    """
    SELECT
      year,
      COUNT(*) AS movie_count,
      ROUND(AVG(rating_score), 3) AS avg_rating,
      ROUND(AVG(rating_count), 1) AS avg_rating_count
    FROM movies
    WHERE year BETWEEN 2000 AND 2021
    GROUP BY year
    ORDER BY year
    """
)
q3.show(30, truncate=False)

print("\n========== Query 4: Window Function Top Movies In Each Genre ==========")
window_spec = Window.partitionBy("genre").orderBy(col("rating_score").desc(), col("rating_count").desc())
q4 = (
    genre_movies.select("genre", "title", "original_title", "year", "rating_score", "rating_count")
    .withColumn("rank_in_genre", row_number().over(window_spec))
    .filter(col("rank_in_genre") <= 3)
    .orderBy("genre", "rank_in_genre")
)
q4.show(60, truncate=26)

spark.stop()
