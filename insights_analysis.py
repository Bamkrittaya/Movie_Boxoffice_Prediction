# ------------------------------------------------------------
# insights_analysis.py
# Generates visual and statistical insights from the trained model
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score
from preprocess_data import prepare_dataset
from train_model import train_and_evaluate
import numpy as np

def analyze_model():
    # Train and get model, test data, and predictions
    model, X_test, y_test, y_pred = train_and_evaluate()

    # Combine results for easy analysis
    df = X_test.copy()
    df["Actual_Revenue"] = y_test
    df["Predicted_Revenue"] = y_pred

    # ------------------------------------------------------------
    # 1️⃣ Key Revenue Drivers
    # ------------------------------------------------------------
    print("\n🔍 KEY REVENUE DRIVERS")
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    print(importance.head(10))  # top 10 features numerically

    plt.figure(figsize=(10,6))
    sns.barplot(data=importance.head(10), x="Coefficient", y="Feature", palette="crest")
    plt.title("Top 10 Features Influencing Revenue")
    plt.tight_layout()
    plt.show()

    print(
        "Interpretation: Larger positive coefficients indicate that those features (e.g., budget, popularity, "
        "or certain genres) strongly contribute to higher predicted revenue."
    )

    # ------------------------------------------------------------
    # 2️⃣ Budget-to-Revenue Relationship
    # ------------------------------------------------------------
    print("\n💰 BUDGET VS REVENUE RELATIONSHIP")

    sns.scatterplot(x="budget", y="Actual_Revenue", data=df, alpha=0.6)
    plt.title("Budget vs Actual Revenue")
    plt.xlabel("Budget (USD)")
    plt.ylabel("Revenue (USD)")
    plt.tight_layout()
    plt.show()

    corr = df["budget"].corr(df["Actual_Revenue"])
    print(f"Correlation between Budget and Revenue: {corr:.2f}")
    print(
        "Interpretation: A strong positive correlation means higher production budgets are generally "
        "linked to higher box office returns, though diminishing returns may occur at very high budgets."
    )

    # ------------------------------------------------------------
    # 3️⃣ Genre and Seasonal Trends
    # ------------------------------------------------------------
    print("\n🎭 GENRE AND SEASONAL TRENDS")

    # Average revenue by genre (only columns that are genre flags)
    genre_columns = [col for col in X_test.columns if col not in ["budget", "popularity", "runtime", "release_month"]]
    genre_revenue = {}
    for genre in genre_columns:
        genre_revenue[genre] = df[df[genre] == 1]["Actual_Revenue"].mean()

    genre_df = pd.DataFrame(list(genre_revenue.items()), columns=["Genre", "Average_Revenue"])
    genre_df = genre_df.sort_values(by="Average_Revenue", ascending=False).head(10)

    plt.figure(figsize=(10,6))
    sns.barplot(data=genre_df, x="Average_Revenue", y="Genre", hue="Genre", palette="mako", legend=False)
    plt.title("Top Performing Genres by Average Revenue")
    plt.tight_layout()
    plt.show()

    # Seasonal trend by release month
    month_revenue = df.groupby("release_month")["Actual_Revenue"].mean()
    plt.figure(figsize=(8,5))
    month_revenue.plot(kind="bar", color="skyblue")
    plt.title("Average Revenue by Release Month")
    plt.ylabel("Average Revenue")
    plt.tight_layout()
    plt.show()

    print(
        "Interpretation: Genres with the highest average revenue (e.g., Action, Adventure) tend to attract larger audiences, "
        "while release months showing higher averages may correspond to holiday or summer periods when audience turnout is greater."
    )

    # ------------------------------------------------------------
    # 4️⃣ Predictive Benchmarking
    # ------------------------------------------------------------
    print("\n📈 PREDICTIVE BENCHMARKING")

    # Example hypothetical movie (pre-release parameters)
    sample_movie = pd.DataFrame([{
        "budget": 80000000,
        "popularity": 25,
        "runtime": 120,
        "release_month": 7,
        # Example: action + adventure
        **{col: (1 if col in ["Action", "Adventure"] else 0) for col in genre_columns}
    }])

    predicted_value = model.predict(sample_movie)[0]
    print(f"Estimated Box Office for Hypothetical Movie: ${predicted_value:,.0f}")

    # Plot where the new prediction stands among historical data
    plt.figure(figsize=(8,5))
    sns.histplot(df["Actual_Revenue"], bins=30, color="lightgray", kde=True)
    plt.axvline(predicted_value, color="red", linestyle="--", linewidth=2)
    plt.title("Predicted Revenue Compared to Historical Distribution")
    plt.xlabel("Revenue (USD)")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.savefig("images/predictive_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        "Interpretation: The red line shows where the predicted revenue for a new film falls relative "
        "to historical box-office performance, providing a benchmark for expected earnings."
    )


    # ------------------------------------------------------------
    # 5️⃣ Comparison Between Movies
    # ------------------------------------------------------------
    print("\n🎬 COMPARISON BETWEEN MOVIES")

    # Compute prediction error
    df["Prediction_Error"] = abs(df["Actual_Revenue"] - df["Predicted_Revenue"])

    # Pick top 10 most accurate predictions
    comparison = df.sort_values("Prediction_Error").head(10)

    # Bar chart: actual vs predicted for those movies
    plt.figure(figsize=(10,6))
    bar_width = 0.4
    x = np.arange(len(comparison))
    plt.bar(x - bar_width/2, comparison["Actual_Revenue"], width=bar_width, label="Actual", color="skyblue")
    plt.bar(x + bar_width/2, comparison["Predicted_Revenue"], width=bar_width, label="Predicted", color="salmon")
    plt.xticks(x, [f"Movie {i+1}" for i in range(len(comparison))], rotation=45)
    plt.ylabel("Revenue (USD)")
    plt.title("Actual vs Predicted Revenue for 10 Most Accurate Movies")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/comparison_movies.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        "Interpretation: This comparison shows how closely the model matches reality for well-predicted films. "
        "Smaller gaps indicate higher model reliability for similar future projects."
    )


    # ------------------------------------------------------------
    # 6️⃣ Marketing and Casting Insights
    # ------------------------------------------------------------
    print("\n📢 MARKETING AND CASTING INSIGHTS")

    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="popularity", y="Actual_Revenue", alpha=0.6, color="purple")
    plt.title("Popularity vs. Actual Revenue")
    plt.xlabel("Popularity Score (Proxy for Marketing/Cast Reach)")
    plt.ylabel("Revenue (USD)")
    plt.tight_layout()
    plt.savefig("images/marketing_casting.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        "Interpretation: The positive trend between popularity and revenue suggests that "
        "movies with higher pre-release attention or well-known casts tend to earn more. "
        "This supports marketing and casting strategies focused on visibility and engagement."
    )


if __name__ == "__main__":
    analyze_model()
