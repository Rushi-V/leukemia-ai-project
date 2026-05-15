import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("biased_leukemia_dataset.csv")

print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# =============================================================================
# 2. DROP NON-PREDICTIVE COLUMNS
# =============================================================================
# Patient_ID is just a row identifier — it carries no medical signal and
# would only add noise if left in as a feature.

df = df.drop(columns=["Patient_ID"])

# =============================================================================
# 3. PREPROCESSING & CLEANING
# =============================================================================

# --- 3a. Standardise text formatting ---
# Strip whitespace and apply Title Case to all string columns so that
# mappings below match reliably (e.g. "male" → "Male", " Yes " → "Yes").
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip().str.title()

# --- 3b. Binary encode Gender ---
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Warn if any values failed to map (e.g. unexpected strings like "Other")
gender_nulls = df["Gender"].isna().sum()
if gender_nulls > 0:
    print(f"WARNING: {gender_nulls} rows have unmapped Gender values — they will be dropped")

# --- 3c. Ordinal encode Socioeconomic_Status ---
# Low < Medium < High represents a meaningful order, so ordinal encoding
# is more appropriate than one-hot encoding here.
df["Socioeconomic_Status"] = df["Socioeconomic_Status"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

ses_nulls = df["Socioeconomic_Status"].isna().sum()
if ses_nulls > 0:
    print(f"WARNING: {ses_nulls} rows have unmapped Socioeconomic_Status values — they will be dropped")

# --- 3d. Binary encode all Yes/No columns ---
# Smoking_Status and Alcohol_Consumption are now included — previously
# missing from this list, which caused them to fall through to get_dummies
# and produce inconsistent column names like "Smoking_Status_Yes".
yes_no_cols = [
    "Infection_History",
    "Immune_Disorders",
    "Chronic_Illness",
    "Genetic_Mutation",
    "Family_History",
    "Radiation_Exposure",
    "Smoking_Status",        # FIX: was missing
    "Alcohol_Consumption",   # FIX: was missing
]

for col in yes_no_cols:
    df[col] = df[col].map({"No": 0, "Yes": 1})
    nulls = df[col].isna().sum()
    if nulls > 0:
        print(f"WARNING: {nulls} rows have unmapped values in '{col}' — they will be dropped")

# --- 3e. One-hot encode remaining categorical columns ---
# This handles: Country, Ethnicity, Urban_Rural.
# drop_first=True removes one dummy column per group to avoid multicollinearity.
# Note: Country will add ~20 columns. If geographic origin is not clinically
# meaningful for your use case, consider: df = df.drop(columns=["Country"])
target = "Leukemia_Status"

remaining_text_cols = [
    col for col in df.select_dtypes(include=["object"]).columns
    if col != target
]
df = pd.get_dummies(df, columns=remaining_text_cols, drop_first=True)

# --- 3f. Convert any boolean columns (from get_dummies) to integers ---
bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

# --- 3g. Drop rows where any mapping failed (NaN values) ---
rows_before = len(df)
df = df.dropna()
rows_dropped = rows_before - len(df)
if rows_dropped > 0:
    print(f"INFO: Dropped {rows_dropped} rows due to failed mappings or missing values")

print(f"Clean dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# =============================================================================
# 4. ENCODE TARGET COLUMN
# =============================================================================
# Using LabelEncoder makes the class mapping explicit so we know for
# certain which index corresponds to "Positive" (leukemia present).
# Without this, the RF still works but predict_proba[:, 1] is ambiguous.

le = LabelEncoder()
y = le.fit_transform(df[target])

print(f"\nTarget class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Class distribution  — Negative: {(y == 0).sum():,} | Positive: {(y == 1).sum():,}")
print(f"Imbalance ratio     — {(y == 0).sum() / len(y) * 100:.1f}% Negative / {(y == 1).sum() / len(y) * 100:.1f}% Positive")

# =============================================================================
# 5. SPLIT DATA
# =============================================================================
# stratify=y ensures both train and test sets preserve the ~85/15 class ratio.

x = df.drop(target, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain set: {len(x_train):,} rows | Test set: {len(x_test):,} rows")

# NOTE: StandardScaler has been removed. Random Forests are tree-based and
# completely scale-invariant — scaling has zero effect on predictions and
# only adds processing time on a dataset this large (143k rows).

# =============================================================================
# 6. TRAIN RANDOM FOREST
# =============================================================================
# n_estimators=100  — 100 decision trees in the ensemble
# class_weight="balanced" — automatically up-weights the minority class
#   (Positive / leukemia present) to compensate for the 85/15 imbalance.
# random_state=42   — ensures reproducible results

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(x_train, y_train)

# =============================================================================
# 7. EVALUATE MODEL
# =============================================================================
y_pred = model.predict(x_test)
probs  = model.predict_proba(x_test)[:, 1]  # index 1 = "Positive" (verified above)

print("\n--- Random Forest Model Results ---")

# AUC (Area Under the ROC Curve) is the primary metric here.
# With an 85/15 class split, raw accuracy is misleading — a model that
# always predicts "Negative" scores 85% accuracy while being useless.
# AUC ranges from 0.5 (random) to 1.0 (perfect) and is imbalance-robust.
auc = roc_auc_score(y_test, probs)
print(f"AUC Score : {auc:.4f}")

# Classification report shows per-class precision, recall, and F1.
# For a cancer screening tool, recall for the Positive class is critical —
# a missed diagnosis (false negative) is worse than a false alarm.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Probability range sanity check — confirm the model is not collapsing
# all predictions to one extreme (0.0 or 1.0).
print(f"Probability range: {probs.min():.4f} – {probs.max():.4f}")
print(f"First 10 predicted probabilities (leukemia risk %):")
print([f"{p * 100:.1f}%" for p in probs[:10]])

# =============================================================================
# 8. CROSS-VALIDATION
# =============================================================================
# A single 80/20 split can produce a lucky or unlucky result. Stratified
# 5-fold CV trains and evaluates on 5 different splits, giving a more
# reliable estimate of real-world performance and its variance.

print("\n--- 5-Fold Cross-Validation (AUC) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
print(f"Per-fold AUC : {[f'{s:.4f}' for s in cv_auc]}")
print(f"Mean AUC     : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================
# Shows which variables contributed most to the model's decisions.
# Higher importance = more influential in predicting leukemia risk.

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature"   : x.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n--- Top 10 Most Influential Features ---")
print(feature_importance_df.head(10).to_string(index=False))

# =============================================================================
# 10. SAVE MODEL ARTIFACTS
# =============================================================================
# Persist the trained model and label encoder so predictions can be made
# on new patients without retraining. Load them later with:
#
#   model = joblib.load("leukemia_model.pkl")
#   le    = joblib.load("leukemia_label_encoder.pkl")
#   risk  = model.predict_proba(new_patient_data)[:, 1] * 100

joblib.dump(model, "leukemia_model.pkl")
joblib.dump(le,    "leukemia_label_encoder.pkl")

print("\nModel saved to: leukemia_model.pkl")
print("Label encoder saved to: leukemia_label_encoder.pkl")
