# ------------------------------------------------------------
# Movie Box Office Prediction Model
# ------------------------------------------------------------
# Author: Krittaya Kruapat (Alice)
# Goal: Predict a movie's box office revenue using pre-release data
# ------------------------------------------------------------

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ast

print("✅ Libraries loaded successfully!")

# ------------------------------------------------------------
# Step 2: Load datasets (Kaggle CSV files)
# ------------------------------------------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
print("✅ Datasets loaded successfully!")

# ------------------------------------------------------------
# Step 3: Merge both datasets on movie ID
# ------------------------------------------------------------
data = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
print(f"✅ Merged dataset shape: {data.shape}")

# ------------------------------------------------------------
# Step 4: Select pre-release features and target
# ------------------------------------------------------------
useful_cols = ["budget", "popularity", "runtime", "release_date", "genres", "revenue"]
df = data[useful_cols].copy()

# ------------------------------------------------------------
# Step 5: Clean missing data and extract month
# ------------------------------------------------------------
df["budget"] = df["budget"].replace(0, np.nan).fillna(df["budget"].median())
df["popularity"] = df["popularity"].fillna(df["popularity"].median())
df["runtime"] = df["runtime"].fillna(df["runtime"].median())

# Convert release_date → month
df["release_month"] = pd.to_datetime(df["release_date"], errors="coerce").dt.month
df.drop(columns=["release_date"], inplace=True)

# ------------------------------------------------------------
# Step 6: Convert genres (list of dicts → dummy columns)
# ------------------------------------------------------------
def extract_genres(x):
    try:
        return [g["name"] for g in ast.literal_eval(x)]
    except Exception:
        return []

df["genres"] = df["genres"].apply(extract_genres)
all_genres = list(set([g for sub in df["genres"] for g in sub]))

for genre in all_genres:
    df[genre] = df["genres"].apply(lambda x: 1 if genre in x else 0)

df.drop(columns=["genres"], inplace=True)

# ------------------------------------------------------------
# Step 7: Split data into features (X) and target (y)
# ------------------------------------------------------------
# Remove or fill any remaining NaN values before splitting
df = df.dropna(subset=["revenue"])       # drop rows with missing target
df = df.fillna(0)                        # fill remaining NaN in features with 0

X = df.drop(columns=["revenue"])
y = df["revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data split into training and testing sets")
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# ------------------------------------------------------------
# Step 8: Train Linear Regression model
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model training complete!")

# ------------------------------------------------------------
# Step 9: Evaluate model performance
# ------------------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R² Score: {r2:.3f}")

# ------------------------------------------------------------
# Step 10: Visualize actual vs predicted revenues
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("🎬 Actual vs Predicted Box Office Revenue")
plt.tight_layout()
plt.show()

print("\n✅ Box Office Prediction pipeline complete!")


# ------------------------------------------------------------
# EXTRA INSIGHTS & ANALYSIS
# ------------------------------------------------------------

# 1. Feature importance (key revenue drivers)
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(importance.head(10))


# 2. Budget vs Revenue relationship
sns.scatterplot(x='budget', y='revenue', data=df, alpha=0.5)
plt.title("Budget vs Revenue")
plt.show()


# 3. Genre and Seasonal trends
genre_revenue = df.groupby('Action')['revenue'].mean().sort_values(ascending=False)
print(genre_revenue.head())

month_revenue = df.groupby('release_month')['revenue'].mean()
month_revenue.plot(kind='bar', title='Average Revenue by Release Month')
plt.show()

# 4. Samples
sample = pd.DataFrame([{
    'budget': 80000000,
    'popularity': 25,
    'runtime': 120,
    'release_month': 7,
    'Action': 1, 'Comedy': 0, 'Drama': 0, 'Adventure': 1,  # etc.
}])
pred = model.predict(sample)[0]
print(f"Predicted box office: ${pred:,.0f}")
