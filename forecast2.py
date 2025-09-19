import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("frp_poll_data.csv")  # Replace with your actual file
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df["date_ordinal"] = df["date"].map(lambda x: x.toordinal())

X = df[["date_ordinal"]]
y = df["FrP"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Optuna objective function
def objective(trial):
    degree = trial.suggest_int("degree", 1, 10)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Best parameters
print("Best RMSE:", study.best_value)
print("Best degree:", study.best_params["degree"])

# Train final model with best degree
best_degree = study.best_params["degree"]
final_model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
final_model.fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["date"], y=df["FrP"], label="Actual Poll %", color="blue")
sns.lineplot(
    x=df["date"].iloc[len(X_train) :],
    y=y_final_pred.flatten(),
    label=f"Predicted (Poly deg={best_degree})",
    color="darkgreen",
)
plt.title(f"FrP Poll Prediction - Polynomial Regression (Degree {best_degree})")
plt.xlabel("Date")
plt.ylabel("Polling Percentage")
plt.legend()
plt.tight_layout()
plt.show()
