# ------------------------------------------------------------
# preprocess_data.py
# Loads and cleans the TMDB 5000 dataset for box office prediction
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import ast

def prepare_dataset(movies_path="tmdb_5000_movies.csv", credits_path="tmdb_5000_credits.csv"):
    """Load and preprocess the TMDB datasets."""
    
    # Load datasets
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    data = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")

    # Select relevant columns
    useful_cols = ["budget", "popularity", "runtime", "release_date", "genres", "revenue"]
    df = data[useful_cols].copy()

    # Fill missing numeric values
    df["budget"] = df["budget"].replace(0, np.nan).fillna(df["budget"].median())
    df["popularity"] = df["popularity"].fillna(df["popularity"].median())
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())

    # Convert release_date → month
    df["release_month"] = pd.to_datetime(df["release_date"], errors="coerce").dt.month
    df.drop(columns=["release_date"], inplace=True)

    # Convert genres JSON to dummy columns
    def extract_genres(x):
        try:
            return [g["name"] for g in ast.literal_eval(x)]
        except Exception:
            return []
    
    df["genres"] = df["genres"].apply(extract_genres)
    all_genres = list(set([g for sublist in df["genres"] for g in sublist]))
    for genre in all_genres:
        df[genre] = df["genres"].apply(lambda x: 1 if genre in x else 0)
    df.drop(columns=["genres"], inplace=True)

    # Remove rows with missing target and fill remaining NaNs
    df = df.dropna(subset=["revenue"])
    df = df.fillna(0)

    # Split features and target
    X = df.drop(columns=["revenue"])
    y = df["revenue"]

    print(f"✅ Dataset ready: {df.shape[0]} movies, {X.shape[1]} features")
    return X, y, all_genres

# Quick test
if __name__ == "__main__":
    X, y, genres = prepare_dataset()
    print("Sample features:", X.columns[:10])
