import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    isnan,
    lit,
    round as spark_round,
    sum as spark_sum,
    trim,
    when,
)
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, StringType


spark = SparkSession.builder.appName("DoubanMovieCleaning").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

input_path = os.getenv("DOUBAN_INPUT", "local:///opt/spark/work/douban_movies.csv")
output_path = os.getenv("DOUBAN_OUTPUT", "")

print("========== A-1 Douban Movie Data Cleaning ==========")
print("Input path:", input_path)

df = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("quote", '"')
    .option("escape", '"')
    .csv(input_path)
)

print("\n========== Schema ==========")
df.printSchema()

print("\n========== First 5 Rows ==========")
df.show(5, truncate=False)

before_count = df.count()
print(f"\nRows before cleaning: {before_count}")

missing_exprs = []
for field in df.schema.fields:
    name = field.name
    dtype = field.dataType
    if isinstance(dtype, StringType):
        missing_cond = col(name).isNull() | (trim(col(name)) == "")
    elif isinstance(dtype, (DoubleType, FloatType)):
        missing_cond = col(name).isNull() | isnan(col(name))
    else:
        missing_cond = col(name).isNull()
    missing_exprs.append(spark_sum(when(missing_cond, 1).otherwise(0)).alias(name))

missing_counts = df.agg(*missing_exprs)
missing_rows = []
for row in missing_counts.collect():
    row_dict = row.asDict()
    for column_name, missing_count in row_dict.items():
        missing_rows.append(
            (
                column_name,
                int(missing_count),
                float(missing_count) / before_count if before_count else 0.0,
            )
        )

missing_df = spark.createDataFrame(
    missing_rows, ["column_name", "missing_count", "missing_ratio"]
).orderBy(col("missing_ratio").desc())

print("\n========== Missing Value Ratio ==========")
missing_df.withColumn("missing_ratio", spark_round(col("missing_ratio"), 4)).show(
    50, truncate=False
)

print("\n========== Cleaning Strategies ==========")
print("Strategy 1: drop rows where genres, rating_score or year is missing.")
print("Strategy 2: fill summary, directors and countries with default text values.")

cleaned = (
    df.dropna(subset=["genres", "rating_score", "year"])
    .fillna(
        {
            "summary": "暂无简介",
            "directors": "未知导演",
            "countries": "未知地区",
        }
    )
    .withColumn("year", col("year").cast(IntegerType()))
    .withColumn("rating_score", col("rating_score").cast(DoubleType()))
    .withColumn("rating_count", col("rating_count").cast(LongType()))
    .withColumn("collect_count", col("collect_count").cast(LongType()))
)

after_count = cleaned.count()
print(f"\nRows after cleaning: {after_count}")
print(f"Removed rows: {before_count - after_count}")

print("\n========== Cleaned First 5 Rows ==========")
cleaned.show(5, truncate=False)

print("\n========== Numeric Statistics ==========")
cleaned.select("year", "rating_score", "rating_count", "collect_count").describe().show(
    truncate=False
)

print("\n========== Non-null Counts After Cleaning ==========")
cleaned.select(
    [count(when(col(c).isNotNull(), c)).alias(c) for c in cleaned.columns]
).show(truncate=False)

if output_path:
    print("\nOutput path:", output_path)
    (
        cleaned.write.mode("overwrite")
        .option("header", "true")
        .csv(output_path)
    )
else:
    print("\nNo DOUBAN_OUTPUT configured. Skip writing cleaned dataset.")

spark.stop()
