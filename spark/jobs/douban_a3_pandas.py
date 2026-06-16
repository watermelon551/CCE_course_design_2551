import os
import time
from pathlib import Path

import pandas as pd


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "douban_movies.csv"


data_path = Path(os.getenv("DOUBAN_CSV", default_data_path()))

print("========== A-3 Pandas Genre GROUP BY ==========")
print("Execution mode: Pandas single machine")
print("Input path:", data_path)

start = time.perf_counter()

df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

movies = df.dropna(subset=["genres", "rating_score", "year"]).copy()
genre_movies = movies.assign(genre=movies["genres"].str.split("/")).explode("genre")
genre_movies["genre"] = genre_movies["genre"].astype(str).str.strip()
genre_movies = genre_movies[genre_movies["genre"] != ""]

result = (
    genre_movies.groupby("genre")
    .agg(
        movie_count=("movie_id", "count"),
        avg_rating=("rating_score", "mean"),
        avg_rating_count=("rating_count", "mean"),
    )
    .reset_index()
)
result = result[result["movie_count"] >= 100]
result = result.sort_values("movie_count", ascending=False).head(15)
result["avg_rating"] = result["avg_rating"].round(3)
result["avg_rating_count"] = result["avg_rating_count"].round(1)

elapsed = time.perf_counter() - start

print(result.to_string(index=False))
print(f"PANDAS_ELAPSED_SECONDS: {elapsed:.6f}")
