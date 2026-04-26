import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("biased_leukemia_dataset.csv")

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

x = df.drop(target, axis=1)
y = df[target]

print("Text columns in x:")
print(x.dtypes[x.dtypes == "object"])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

probs = model.predict_proba(x_test)[:, 1]
print("Probability range:", probs.min(), probs.max())
print("First 20 probabilities:", probs[:20])