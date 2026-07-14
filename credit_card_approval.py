# =============================================================================
# Credit Card Fraud / Approval Prediction
# Dataset: Kaggle Credit Card Fraud Detection (creditcard.csv)
# Features: PCA-transformed V1–V28, Amount, Time → predict Class (0=legit, 1=fraud)
# =============================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "creditcard.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ===========================================================================
# EPIC 2 – DATA LOADING & EXPLORATION
# ===========================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the credit-card dataset and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded → shape: {df.shape}")
    print(df.head(3))
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nBasic info:")
    df.info()
    return df


def univariate_analysis(df: pd.DataFrame) -> None:
    """Story 3 – Univariate analysis: distributions of key columns."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Class distribution
    class_counts = df["Class"].value_counts()
    axes[0].bar(["Legitimate (0)", "Fraud (1)"],
                class_counts.values,
                color=["steelblue", "crimson"])
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 200, str(val),
                     ha="center", va="bottom", fontsize=10)

    # Amount distribution (log scale)
    axes[1].hist(df["Amount"], bins=60, color="teal", edgecolor="white")
    axes[1].set_title("Transaction Amount Distribution")
    axes[1].set_xlabel("Amount")
    axes[1].set_ylabel("Frequency")

    # Time distribution
    axes[2].hist(df["Time"], bins=48, color="darkorange", edgecolor="white")
    axes[2].set_title("Transaction Time Distribution")
    axes[2].set_xlabel("Seconds from first transaction")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "univariate_analysis.png"), dpi=120)
    plt.close()
    print("[INFO] Saved → univariate_analysis.png")


def multivariate_analysis(df: pd.DataFrame) -> None:
    """Story 4 – Correlation heatmap of all numeric features."""
    corr = df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.3,
                annot=False, fmt=".1f", square=True)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=100)
    plt.close()
    print("[INFO] Saved → correlation_heatmap.png")


def descriptive_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Story 5 – Descriptive statistics."""
    stats = df.describe().T
    print("\nDescriptive Statistics:\n", stats)
    return stats


# ===========================================================================
# EPIC 3 – PRE-PROCESSING
# ===========================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Story 1 – Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[INFO] Duplicates removed: {before - after} | Remaining rows: {after}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Story 2 – Detect and handle missing values."""
    missing = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100
    print("\nMissing values per column:")
    print(pd.concat([missing, missing_pct], axis=1,
                    keys=["Count", "Percent(%)"])
            .query("Count > 0"))

    # Fill any numeric NaN with column median (safe fallback)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print("[INFO] Missing values handled.")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Story 4 – Feature engineering: normalise Amount and Time."""
    scaler = StandardScaler()
    df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_Scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)
    print("[INFO] Feature engineering complete.")
    return df


def prepare_features(df: pd.DataFrame):
    """Split into X (features) and y (target) then train/test split."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Class balance (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# EPIC 4 – MODEL BUILDING
# ===========================================================================

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Helper – print metrics and return summary dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Fraud"]))
    print(f"  F1-Score : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC  : {auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(PLOT_DIR, f"cm_{safe_name}.png"), dpi=100)
    plt.close()

    return {"model": name, "f1": f1, "auc": auc, "estimator": model}


def train_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    """Story 1 – Logistic Regression."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model("Logistic Regression", model, X_test, y_test)


def train_decision_tree(X_train, X_test, y_train, y_test) -> dict:
    """Story 3 – Decision Tree."""
    model = DecisionTreeClassifier(max_depth=10, class_weight="balanced",
                                   random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model("Decision Tree", model, X_test, y_test)


def train_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """Story 2 – Random Forest."""
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model("Random Forest", model, X_test, y_test)


def train_gradient_boosting(X_train, X_test, y_train, y_test) -> dict:
    """Bonus – Gradient Boosting."""
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model("Gradient Boosting", model, X_test, y_test)


def compare_models(results: list) -> str:
    """Story 4 – Bar chart comparing F1 scores; return name of best model."""
    names = [r["model"] for r in results]
    f1s   = [r["f1"]    for r in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, f1s,
                   color=["steelblue", "coral", "seagreen", "mediumpurple"])
    plt.ylim(0, 1.05)
    plt.title("Model Comparison – F1-Score (Fraud Class)")
    plt.ylabel("F1-Score")
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=120)
    plt.close()
    print("[INFO] Saved → model_comparison.png")

    best = max(results, key=lambda r: r["f1"])
    print(f"\n[RESULT] Best model: {best['model']} (F1={best['f1']:.4f})")
    return best


def save_model(best: dict) -> None:
    """Persist the best model with pickle."""
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(best["estimator"], f)
    print(f"[INFO] Model saved → {path}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print("\n" + "="*60)
    print("  CREDIT CARD APPROVAL / FRAUD PREDICTION PIPELINE")
    print("="*60 + "\n")

    # -- Load
    df = load_data(DATA_PATH)

    # -- Explore
    univariate_analysis(df)
    multivariate_analysis(df)
    descriptive_analysis(df)

    # -- Pre-process
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = feature_engineering(df)

    # -- Split
    X_train, X_test, y_train, y_test = prepare_features(df)

    # -- Train all models
    results = []
    results.append(train_logistic_regression(X_train, X_test, y_train, y_test))
    results.append(train_decision_tree(X_train, X_test, y_train, y_test))
    results.append(train_random_forest(X_train, X_test, y_train, y_test))
    results.append(train_gradient_boosting(X_train, X_test, y_train, y_test))

    # -- Compare & Save
    best = compare_models(results)
    save_model(best)

    print("\n[DONE] Pipeline complete. See models/ and static/plots/ for outputs.\n")


if __name__ == "__main__":
    main()
