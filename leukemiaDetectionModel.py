import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
df = pd.read_csv("biased_leukemia_dataset.csv")

# 2. Preprocessing & Cleaning
# Clean all text columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip().str.title()

# Manual ordered/binary mappings
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Socioeconomic_Status"] = df["Socioeconomic_Status"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

yes_no_cols = [
    "Infection_History",
    "Immune_Disorders",
    "Chronic_Illness",
    "Genetic_Mutation",
    "Family_History",
    "Radiation_Exposure"
]

for col in yes_no_cols:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# One-hot encode remaining categorical columns except target
target = "Leukemia_Status"

remaining_text_cols = [
    col for col in df.select_dtypes(include=["object"]).columns
    if col != target
]

df = pd.get_dummies(df, columns=remaining_text_cols, drop_first=True)

# Convert True/False to 0/1
bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

# Drop rows with failed mappings/missing values
df = df.dropna()

# 3. Split Data
x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scaling (Optional for RF, but good for consistency)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Random Forest Integration
# n_estimators: number of trees in the forest
# class_weight="balanced": handles the "biased" nature of the dataset
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(x_train, y_train)

# 6. Evaluation
y_pred = model.predict(x_test)

print("--- Random Forest Model Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Probability Analysis
probs = model.predict_proba(x_test)[:, 1]
print("\nProbability range:", probs.min(), "-", probs.max())
print("First 10 predicted probabilities:", probs[:10])

# 8. Feature Importance (Bonus)
# This identifies which variables had the most impact on the prediction
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 5 Most Influential Features:")
print(feature_importance_df.head(5))
