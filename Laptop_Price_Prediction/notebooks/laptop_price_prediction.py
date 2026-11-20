"""
================================================================
 LAPTOP PRICE PREDICTION USING MACHINE LEARNING
================================================================
 A complete, beginner-friendly ML pipeline covering:
   Part  1 – Problem Definition
   Part  2 – Dataset Understanding
   Part  3 – Data Preprocessing
   Part  4 – Exploratory Data Analysis (EDA)
   Part  5 – Feature & Target Selection
   Part  6 – Train-Test Split
   Part  7 – Model Selection
   Part  8 – Model Training
   Part  9 – Model Evaluation
   Part 10 – Model Visualisation
   Part 11 – Model Improvement (Hyperparameter Tuning)
   Part 12 – Save Final Model

 Author : ML Project Pipeline
 Date   : 2026-11-13
================================================================
"""

# ──────────────────────────────────────────────────────────
# 📦  Imports
# ──────────────────────────────────────────────────────────
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ──────────────────────────────────────────────────────────
# 📂  Paths (relative to project root)
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "laptop_data.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ╔════════════════════════════════════════════════════════╗
# ║  PART 1 – PROBLEM DEFINITION                          ║
# ╚════════════════════════════════════════════════════════╝
print("=" * 60)
print("  PART 1 – PROBLEM DEFINITION")
print("=" * 60)
print("""
PROBLEM STATEMENT
-----------------
Laptops come in a wide variety of configurations and prices.
Predicting a fair market price from specifications helps both
buyers and sellers make informed decisions.

OBJECTIVE
---------
Build a Regression model that predicts the price of a laptop
(in Indian Rupees, ₹) given its specifications such as:
  • Company / Brand
  • Laptop Type (Notebook, Gaming, Ultrabook …)
  • RAM, Storage (HDD / SSD)
  • Processor (CPU brand & model)
  • GPU (Graphics card brand)
  • Screen Size & Resolution
  • Operating System
  • Weight

We will train and compare several regression models and select
the best-performing one.
""")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 2 – DATASET UNDERSTANDING                       ║
# ╚════════════════════════════════════════════════════════╝
print("=" * 60)
print("  PART 2 – DATASET UNDERSTANDING")
print("=" * 60)

# Load the dataset
df = pd.read_csv(DATA_PATH)

print(f"\n📐  Dataset Shape: {df.shape}")
print(f"     → {df.shape[0]} rows  ×  {df.shape[1]} columns\n")

print("📋  Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"     {i:>2}. {col}")

print("\n🔝  First 5 Rows:")
print(df.head().to_string(index=False))

print("\n📊  Data Types:")
print(df.dtypes.to_string())

print("\n❓  Missing Values:")
missing = df.isnull().sum()
print(missing.to_string())
print(f"\n     Total missing values: {missing.sum()}")

print("\n📈  Statistical Summary:")
print(df.describe().round(2).to_string())


# ╔════════════════════════════════════════════════════════╗
# ║  PART 3 – DATA PREPROCESSING                          ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 3 – DATA PREPROCESSING")
print("=" * 60)

# ── 3.1  Remove duplicates ───────────────────────────────
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"\n🔁  Duplicates removed: {duplicates_before}")
print(f"     Rows after removal: {len(df)}")

# ── 3.2  Handle missing values ───────────────────────────
# Fill numeric NaNs with median; categorical NaNs with mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
print("✅  Missing values handled.")


# ── 3.3  Feature Engineering Helper Functions ────────────

def extract_cpu_brand(cpu_name: str) -> str:
    """
    Simplify CPU names into broad categories.

    Examples
    --------
    >>> extract_cpu_brand("Intel Core i7 7700HQ 2.8GHz")
    'Intel Core i7'
    """
    cpu = str(cpu_name).lower()
    if "intel core i7" in cpu:
        return "Intel Core i7"
    elif "intel core i5" in cpu:
        return "Intel Core i5"
    elif "intel core i3" in cpu:
        return "Intel Core i3"
    elif "amd" in cpu:
        return "AMD Processor"
    else:
        return "Other"


def extract_hdd(memory: str) -> int:
    """
    Extract the HDD capacity in GB from the Memory column.

    Examples
    --------
    >>> extract_hdd("256GB SSD + 1TB HDD")
    1000
    """
    hdd = 0
    mem = str(memory)
    if "HDD" in mem:
        parts = mem.split("+")
        for part in parts:
            if "HDD" in part:
                part = part.strip()
                match = re.search(r"(\d+)", part)
                if match:
                    val = int(match.group())
                    hdd = val * 1000 if "TB" in part else val
    return hdd


def extract_ssd(memory: str) -> int:
    """
    Extract the SSD capacity in GB from the Memory column.

    Examples
    --------
    >>> extract_ssd("256GB SSD + 1TB HDD")
    256
    """
    ssd = 0
    mem = str(memory)
    if "SSD" in mem:
        parts = mem.split("+")
        for part in parts:
            if "SSD" in part:
                part = part.strip()
                match = re.search(r"(\d+)", part)
                if match:
                    val = int(match.group())
                    ssd = val * 1000 if "TB" in part else val
    return ssd


def categorize_os(os_name: str) -> str:
    """
    Simplify OS names into three categories.

    Examples
    --------
    >>> categorize_os("Windows 10")
    'Windows'
    """
    os_name = str(os_name).lower()
    if "windows" in os_name:
        return "Windows"
    elif "mac" in os_name:
        return "Mac"
    else:
        return "Linux/Other"


# ── 3.4  Apply Feature Engineering ───────────────────────
print("\n🔧  Engineering features …")

# 3.4.1 – Extract numeric RAM
df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype(int)
print(f"     ✓ Ram converted to int  (unique: {sorted(df['Ram'].unique())})")

# 3.4.2 – Extract numeric Weight
df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype(float)
print(f"     ✓ Weight converted to float")

# 3.4.3 – Touchscreen flag
df["Touchscreen"] = df["ScreenResolution"].apply(
    lambda x: 1 if "Touchscreen" in str(x) else 0
)
print(f"     ✓ Touchscreen flag  (1: {df['Touchscreen'].sum()}, 0: {(~df['Touchscreen'].astype(bool)).sum()})")

# 3.4.4 – IPS Panel flag
df["IPS"] = df["ScreenResolution"].apply(
    lambda x: 1 if "IPS" in str(x) else 0
)
print(f"     ✓ IPS flag  (1: {df['IPS'].sum()}, 0: {(~df['IPS'].astype(bool)).sum()})")

# 3.4.5 – PPI (Pixels Per Inch)
resolution = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)")
df["X_res"] = resolution[0].astype(int)
df["Y_res"] = resolution[1].astype(int)
df["PPI"] = (
    ((df["X_res"] ** 2 + df["Y_res"] ** 2) ** 0.5) / df["Inches"]
).round(2)
print(f"     ✓ PPI calculated  (mean: {df['PPI'].mean():.1f})")

# 3.4.6 – CPU brand
df["Cpu_brand"] = df["Cpu"].apply(extract_cpu_brand)
print(f"     ✓ Cpu_brand  (unique: {df['Cpu_brand'].unique().tolist()})")

# 3.4.7 – HDD & SSD
df["HDD"] = df["Memory"].apply(extract_hdd)
df["SSD"] = df["Memory"].apply(extract_ssd)
print(f"     ✓ HDD extracted  (unique GB: {sorted(df['HDD'].unique())})")
print(f"     ✓ SSD extracted  (unique GB: {sorted(df['SSD'].unique())})")

# 3.4.8 – GPU brand
df["Gpu_brand"] = df["Gpu"].apply(lambda x: str(x).split()[0])
print(f"     ✓ Gpu_brand  (unique: {df['Gpu_brand'].unique().tolist()})")

# 3.4.9 – OS category
df["OpSys"] = df["OpSys"].apply(categorize_os)
print(f"     ✓ OpSys simplified  (unique: {df['OpSys'].unique().tolist()})")

# ── 3.5  Drop original columns no longer needed ─────────
drop_cols = ["ScreenResolution", "Cpu", "Memory", "Gpu", "X_res", "Y_res", "Inches"]
df.drop(columns=drop_cols, inplace=True)
print(f"\n🗑️   Dropped columns: {drop_cols}")
print(f"📐  New shape: {df.shape}")
print(f"\n📋  Processed columns: {df.columns.tolist()}")
print(df.head().to_string(index=False))


# ╔════════════════════════════════════════════════════════╗
# ║  PART 4 – EXPLORATORY DATA ANALYSIS (EDA)             ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 4 – EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)
print("  Saving plots to:", PLOTS_DIR)

# ── 4.1  Distribution of Laptop Prices ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["Price"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Laptop Prices")
axes[0].set_xlabel("Price (₹)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log(df["Price"]), bins=40, color="salmon", edgecolor="white")
axes[1].set_title("Distribution of log(Price)")
axes[1].set_xlabel("log(Price)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_price_distribution.png"), dpi=150)
plt.close()
print("  ✓ 01_price_distribution.png saved")

# ── 4.2  Average Price by Company ────────────────────────
avg_price_company = (
    df.groupby("Company")["Price"]
    .mean()
    .sort_values(ascending=True)
)
fig, ax = plt.subplots(figsize=(10, 6))
avg_price_company.plot(kind="barh", color="teal", edgecolor="white", ax=ax)
ax.set_title("Average Laptop Price by Company")
ax.set_xlabel("Average Price (₹)")
ax.set_ylabel("Company")
for i, v in enumerate(avg_price_company):
    ax.text(v + 500, i, f"₹{v:,.0f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_avg_price_by_company.png"), dpi=150)
plt.close()
print("  ✓ 02_avg_price_by_company.png saved")

# ── 4.3  RAM vs Price ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="Ram", y="Price", data=df, palette="viridis", ax=ax)
ax.set_title("RAM vs Laptop Price")
ax.set_xlabel("RAM (GB)")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_ram_vs_price.png"), dpi=150)
plt.close()
print("  ✓ 03_ram_vs_price.png saved")

# ── 4.4  Processor Type vs Price ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="Cpu_brand", y="Price", data=df, palette="Set2", ax=ax)
ax.set_title("Processor Type vs Laptop Price")
ax.set_xlabel("Processor")
ax.set_ylabel("Price (₹)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_cpu_vs_price.png"), dpi=150)
plt.close()
print("  ✓ 04_cpu_vs_price.png saved")

# ── 4.5  Correlation Heatmap ────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_correlation_heatmap.png"), dpi=150)
plt.close()
print("  ✓ 05_correlation_heatmap.png saved")

# ── 4.6  TypeName vs Price ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="TypeName", y="Price", data=df, palette="pastel", ax=ax)
ax.set_title("Laptop Type vs Price")
ax.set_xlabel("Laptop Type")
ax.set_ylabel("Price (₹)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_type_vs_price.png"), dpi=150)
plt.close()
print("  ✓ 06_type_vs_price.png saved")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 5 – FEATURE & TARGET SELECTION                   ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 5 – FEATURE & TARGET SELECTION")
print("=" * 60)

# Define feature groups
NUM_FEATURES = ["Ram", "Weight", "Touchscreen", "IPS", "PPI", "HDD", "SSD"]
CAT_FEATURES = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "OpSys"]

print(f"\n  Numeric features  ({len(NUM_FEATURES)}): {NUM_FEATURES}")
print(f"  Categorical features ({len(CAT_FEATURES)}): {CAT_FEATURES}")

# Feature matrix (X) and target vector (y)
X = df[NUM_FEATURES + CAT_FEATURES]
y = np.log(df["Price"])          # log-transform for better regression

print(f"\n  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Target : log(Price)  →  will use np.exp() to get ₹ predictions")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 6 – TRAIN-TEST SPLIT                            ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 6 – TRAIN-TEST SPLIT (80:20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n  Training set : {X_train.shape[0]} samples")
print(f"  Testing  set : {X_test.shape[0]} samples")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 7 & 8 – MODEL SELECTION & TRAINING              ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 7 & 8 – MODEL SELECTION & TRAINING")
print("=" * 60)

# ── Build a reusable preprocessing ColumnTransformer ─────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            CAT_FEATURES,
        ),
    ],
    remainder="drop",
)

# ── Define models to compare ────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

# ── Train each model inside a Pipeline ──────────────────
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    print(f"  ✅  {name:25s} — trained")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 9 – MODEL EVALUATION                            ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 9 – MODEL EVALUATION")
print("=" * 60)

results = []

for name, pipe in trained_pipelines.items():
    y_pred = pipe.predict(X_test)

    # Metrics on log-scale
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Metrics on original ₹ scale (for user-friendly reporting)
    y_test_inr = np.exp(y_test)
    y_pred_inr = np.exp(y_pred)
    mae_inr  = mean_absolute_error(y_test_inr, y_pred_inr)

    results.append({
        "Model": name,
        "R² Score": round(r2, 4),
        "MAE (log)": round(mae, 4),
        "MSE (log)": round(mse, 4),
        "RMSE (log)": round(rmse, 4),
        "MAE (₹)": round(mae_inr, 0),
    })

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
print("\n" + results_df.to_string(index=False))

# ── Automatically select the best model ──────────────────
best_model_name = results_df.iloc[0]["Model"]
best_r2 = results_df.iloc[0]["R² Score"]
print(f"\n🏆  Best model: {best_model_name}  (R² = {best_r2})")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 10 – MODEL VISUALISATION                        ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 10 – MODEL VISUALISATION")
print("=" * 60)

best_pipe = trained_pipelines[best_model_name]
y_pred_best = best_pipe.predict(X_test)

# ── 10.1  Actual vs Predicted Price ─────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
y_test_inr = np.exp(y_test)
y_pred_inr = np.exp(y_pred_best)
ax.scatter(y_test_inr, y_pred_inr, alpha=0.5, edgecolors="w", linewidth=0.5)
max_val = max(y_test_inr.max(), y_pred_inr.max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
ax.set_title(f"Actual vs Predicted Price ({best_model_name})")
ax.set_xlabel("Actual Price (₹)")
ax.set_ylabel("Predicted Price (₹)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_actual_vs_predicted.png"), dpi=150)
plt.close()
print("  ✓ 07_actual_vs_predicted.png saved")

# ── 10.2  Feature Importance (if available) ──────────────
underlying_model = best_pipe.named_steps["model"]
if hasattr(underlying_model, "feature_importances_"):
    # Get feature names from the preprocessor
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(CAT_FEATURES).tolist()
    all_feature_names = NUM_FEATURES + cat_feature_names

    importances = underlying_model.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values()

    # Show top 15 features
    top_features = feat_imp.tail(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features.plot(kind="barh", color="darkorange", edgecolor="white", ax=ax)
    ax.set_title(f"Top 15 Feature Importances ({best_model_name})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_feature_importance.png"), dpi=150)
    plt.close()
    print("  ✓ 08_feature_importance.png saved")

# ── 10.3  Model Comparison Chart ────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
model_names = results_df["Model"].tolist()
r2_scores   = results_df["R² Score"].tolist()
colors = ["gold" if n == best_model_name else "steelblue" for n in model_names]

bars = ax.barh(model_names, r2_scores, color=colors, edgecolor="white")
ax.set_title("Model Comparison – R² Score")
ax.set_xlabel("R² Score")
ax.set_xlim(0, 1.05)
for bar, score in zip(bars, r2_scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_model_comparison.png"), dpi=150)
plt.close()
print("  ✓ 09_model_comparison.png saved")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 11 – MODEL IMPROVEMENT                          ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 11 – MODEL IMPROVEMENT")
print("=" * 60)

# ── 11.1  Hyperparameter Tuning with RandomizedSearchCV ──
print("\n  🔍 Hyperparameter tuning (RandomizedSearchCV) …")
print("     This may take a minute …")

# We tune the best model.  If it is Random Forest, we tune its
# hyper-parameters; otherwise we still tune a Random Forest
# because it generally performs best for this kind of data.
tuning_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42)),
])

param_distributions = {
    "model__n_estimators": [100, 200, 300, 400, 500],
    "model__max_depth": [None, 10, 20, 30, 40],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None],
}

random_search = RandomizedSearchCV(
    tuning_pipe,
    param_distributions=param_distributions,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
random_search.fit(X_train, y_train)

print(f"\n  Best parameters found:")
for k, v in random_search.best_params_.items():
    print(f"     {k}: {v}")
print(f"  Best CV R² Score: {random_search.best_score_:.4f}")

# ── 11.2  Evaluate tuned model on test set ───────────────
tuned_pipe = random_search.best_estimator_
y_pred_tuned = tuned_pipe.predict(X_test)

r2_tuned   = r2_score(y_test, y_pred_tuned)
mae_tuned  = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned  = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)

print(f"\n  📊 Tuned Model Performance on Test Set:")
print(f"     R² Score : {r2_tuned:.4f}")
print(f"     MAE      : {mae_tuned:.4f}")
print(f"     MSE      : {mse_tuned:.4f}")
print(f"     RMSE     : {rmse_tuned:.4f}")

# ── 11.3  Cross-Validation on tuned model ────────────────
print("\n  🔄 5-Fold Cross-Validation on full training data …")
cv_scores = cross_val_score(tuned_pipe, X_train, y_train, cv=5, scoring="r2")
print(f"     CV R² Scores : {np.round(cv_scores, 4).tolist()}")
print(f"     Mean CV R²   : {cv_scores.mean():.4f}")
print(f"     Std  CV R²   : {cv_scores.std():.4f}")

# ── 11.4  Compare original vs tuned ─────────────────────
print(f"\n  📈 Improvement Summary:")
print(f"     Original best R² : {best_r2:.4f}")
print(f"     Tuned R²         : {r2_tuned:.4f}")
improvement = r2_tuned - best_r2
print(f"     Improvement      : {'+' if improvement >= 0 else ''}{improvement:.4f}")

# Use the tuned model if it's better; otherwise keep the original
if r2_tuned >= best_r2:
    final_pipe = tuned_pipe
    final_model_name = "Tuned Random Forest"
    print("     → Using the TUNED model as the final model.")
else:
    final_pipe = best_pipe
    final_model_name = best_model_name
    print("     → Tuning did not improve. Keeping the ORIGINAL model.")


# ╔════════════════════════════════════════════════════════╗
# ║  PART 12 – SAVE FINAL MODEL                           ║
# ╚════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  PART 12 – SAVE FINAL MODEL")
print("=" * 60)

# ── 12.1  Save the trained pipeline ─────────────────────
model_path = os.path.join(MODEL_DIR, "laptop_price_model.pkl")
joblib.dump(final_pipe, model_path)
print(f"\n  💾 Model saved to: {model_path}")

# ── 12.2  Save metadata for the Streamlit app ───────────
metadata = {
    "companies": sorted(df["Company"].unique().tolist()),
    "type_names": sorted(df["TypeName"].unique().tolist()),
    "cpu_brands": sorted(df["Cpu_brand"].unique().tolist()),
    "gpu_brands": sorted(df["Gpu_brand"].unique().tolist()),
    "os_options": sorted(df["OpSys"].unique().tolist()),
    "ram_options": sorted(df["Ram"].unique().tolist()),
    "num_features": NUM_FEATURES,
    "cat_features": CAT_FEATURES,
    "final_model_name": final_model_name,
    "final_r2_score": round(float(r2_tuned if r2_tuned >= best_r2 else best_r2), 4),
}

metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
joblib.dump(metadata, metadata_path)
print(f"  💾 Metadata saved to: {metadata_path}")

# ── 12.3  Quick sanity check ────────────────────────────
print("\n  🧪 Sanity check – loading and predicting with saved model …")
loaded_pipe = joblib.load(model_path)
sample = X_test.iloc[:3]
preds = np.exp(loaded_pipe.predict(sample))
actuals = np.exp(y_test.iloc[:3].values)
for i, (a, p) in enumerate(zip(actuals, preds)):
    print(f"     Sample {i+1}: Actual ₹{a:,.0f}  →  Predicted ₹{p:,.0f}")

print("\n" + "=" * 60)
print("  ✅  ALL DONE!  Pipeline complete.")
print("=" * 60)
print(f"""
  Next steps:
  1. Run the Streamlit app:
       streamlit run app.py

  2. EDA plots are saved in: {PLOTS_DIR}
  3. Trained model is at   : {model_path}
""")
