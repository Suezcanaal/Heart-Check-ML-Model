import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data (From your local file)
# We assume the file is in the same folder as this script
filename = "Heart_disease_cleveland_new.csv"
print(f"Loading data from {filename}...")

# Your file already has headers, so we don't need to manually assign them
df = pd.read_csv(filename)

# 2. Separate features (X) and target (y)
# 'target' is the column we want to predict (0 = No Disease, 1 = Disease)
X = df.drop("target", axis=1)
y = df["target"]

# 3. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost Model
print("Training XGBoost Classifier...")
model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model
joblib.dump(model, "heart_model.pkl")
print("Success! Model saved to 'heart_model.pkl'")