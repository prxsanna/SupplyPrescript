import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# Load dataset
data = pd.read_csv("data/supply_chain_data.csv")

print("Dataset shape:", data.shape)

# Target
target = "delay_days"

# Remove columns that should not be used for prediction
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


# XGBoost regression model
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


print("\nTraining XGBoost model for feature importance...")

# Train
pipeline.fit(X_train, y_train)


# Get transformed feature names
feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()


# Get feature importance
importance_values = pipeline.named_steps[
    "model"
].feature_importances_


# Create feature importance dataframe
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})


# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


print("\n===================================")
print("SUPPLYPRESCRIPT MODEL AUDIT")
print("===================================")

print("\nTop 10 important features:")

print(
    importance_df.head(10).to_string(index=False)
)


# Save results
importance_df.to_csv(
    "ml/feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("ml/feature_importance.csv")