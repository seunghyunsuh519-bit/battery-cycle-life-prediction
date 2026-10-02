from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)


RANDOM_STATE = 42


# =========================================================
# Regression Models
# =========================================================

def get_regression_models():
    """
    Cycle Life Regression 후보 모델 반환
    """

    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# =========================================================
# Classification Models
# =========================================================

def get_classification_models():
    """
    Cycle Life Classification 후보 모델 반환
    """

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# =========================================================
# Feature Columns
# =========================================================

FEATURE_COLS = [
    "mean_QD",
    "std_QD",
    "QD_slope",
    "deltaQ_mean",
    "deltaQ_std",
    "deltaQ_var",
    "deltaQ_min",
    "deltaQ_max",
    "deltaQ_range",
    "mean_IR",
    "IR_slope",
    "mean_Tavg",
    "mean_Tmax",
    "max_Tmax",
    "mean_chargetime",
]


# Classification 기준
CLASS_THRESHOLD = 850