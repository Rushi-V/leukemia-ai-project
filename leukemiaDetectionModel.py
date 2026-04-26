import pandas as pd

df = pd.read_csv("/content/biased_leukemia_dataset.csv")
df.head()
from sklearn.preprocessing import LabelEncoder
categorical_cols = ["Gender", "Infection_History", "Infection_History", "Immune_Disorders", "Chronic_Illness", "Socioeconomic_Status"]
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=["Country"], drop_first=True)
df.head()

country_columns = [x for x in df.columns if x.startswith("Country_")]
x=df[["Age", "Gender", "WBC_Count", "RBC_Count", "Bone_Marrow_Blasts", "BMI", "Socioeconomic_Status", "Infection_History", "Chronic_Illness", "Immune_Disorders"]+country_columns]
y=df["Leukemia_Status"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight = "balanced", max_iter=1000)
model.fit(x_train, y_train)

probs = model.predict_proba(x_test)[:,1]
print(probs.min(), probs.max())
print(probs[:20])

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))