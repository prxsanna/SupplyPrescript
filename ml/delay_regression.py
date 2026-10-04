import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# Load dataset
data = pd.read_csv("data/supply_chain_data.csv")

print("Dataset shape:", data.shape)

# Target
target = "delay_days"

# Features
# We remove:
# shipment_id       -> identifier, not useful for prediction
# delay_days        -> target variable
# is_delayed        -> derived from delay_days, causes target leakage
# actual_lead_time_days -> contains information about the actual outcome
X = data.drop(
    columns=[
        "shipment_id",
        "delay_days",
        "is_delayed",
        "actual_lead_time_days"
    ]
)

y = data[target]

print("\nTarget:", target)

print("\nFeatures used:")
print(X.columns.tolist())

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# XGBoost Regression Model
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)

# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining XGBoost regression model...")

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)
y_pred = y_pred.clip(min=0)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===================================")
print("SUPPLYPRESCRIPT DELAY REGRESSION")
print("===================================")

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Show sample predictions
results = pd.DataFrame({
    "Actual Delay": y_test.values,
    "Predicted Delay": y_pred
})

print("\nSample Predictions:")
print(results.head(10))