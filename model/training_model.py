import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("real_dataset.csv")

print(df.head())

# Features + target
X = df[["mean_area", "std_area", "mean_red", "rbc_count", "pale_ratio"]]
y = df["label"]



# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

acc = model.score(X, y)
print(f"Accuracy: {acc:.2f}")

# Save model
joblib.dump(model, "anemia_model.pkl")

print("✅ Model trained and saved")