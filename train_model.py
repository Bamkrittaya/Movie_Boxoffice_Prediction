# ------------------------------------------------------------
# train_model.py
# Trains and evaluates the box office prediction model
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess_data import prepare_dataset
import numpy as np
import joblib

def train_and_evaluate():
    # Load preprocessed data
    X, y, _ = prepare_dataset()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance:")
    print(f"MAE: {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R² Score: {r2:.3f}")

    # Save model
    joblib.dump(model, "boxoffice_model.pkl")
    print("💾 Model saved as boxoffice_model.pkl")

    # Return everything for further analysis
    return model, X_test, y_test, y_pred

# Quick test
if __name__ == "__main__":
    train_and_evaluate()
