import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import KFold, cross_validate

# TODO: Replace with your actual student ID
STUDENT_ID = 42


def evaluate_strategy(csv_file, strategy):
    """
    Applies encoding and evaluates model performance.
    Strategies: 'ordinal', 'onehot'
    """
    # Load data
    df = pd.read_csv(csv_file)

    target_col = "target"
    features = [c for c in df.columns if c != target_col]

    ordinal_order = ["High School", "Bachelor", "Master", "PhD"]
    natural_order = None

    # Detect if feature has ordinal relationship
    feature_values = df[features[0]].unique()
    if set(feature_values).issubset(set(ordinal_order)):
        natural_order = ordinal_order

    # Apply Transformation
    if strategy == "ordinal":
        # Ordinal Encoding: Use natural order if available, else alphabetical
        if natural_order:
            encoder = OrdinalEncoder(categories=[natural_order])
        else:
            encoder = OrdinalEncoder()

        X = encoder.fit_transform(df[[features[0]]])

    elif strategy == "onehot":
        # One-Hot Encoding: Creates N binary columns (0 or 1)
        encoder = OneHotEncoder(sparse_output=False)
        X = encoder.fit_transform(df[[features[0]]])

    else:
        raise ValueError("Strategy must be 'ordinal' or 'onehot'")

    y = df[target_col]

    # Evaluate using Linear Regression
    cv_method = KFold(n_splits=5, shuffle=True, random_state=STUDENT_ID)
    model = LinearRegression()

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv_method,
        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
        return_train_score=False,
    )

    result = {
        "Strategy": strategy,
        "R²": float(cv_results["test_r2"].mean()),
        "MSE": float(-cv_results["test_neg_mean_squared_error"].mean()),
        "MAE": float(-cv_results["test_neg_mean_absolute_error"].mean()),
    }
    return result


if __name__ == "__main__":
    # R² should be as close to 1 as possible
    # MSE and MAE should be as low as possible
    filename = "data_1.csv"
    # TODO: Keep changing strategy and document results in a table for your report
    # Strategies: ordinal, onehot
    result = evaluate_strategy(csv_file=filename, strategy="onehot")
    print(result)
