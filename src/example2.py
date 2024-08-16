import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# File paths
train_file = r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\train_outcomes.csv"
test_file = r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\test_outcomes.csv"

# Load training data
train_df = pd.read_csv(train_file)

# Preprocess training data
train_df['clientID'] = train_df['clientID'].astype('category')
train_df['resourceID'] = train_df['resourceID'].astype('category')
train_dum = pd.get_dummies(train_df)

# Convert outcome labels from [-1, 0, 1] to [0, 1, 2] if needed
train_dum['outcome'] = train_dum['outcome'].map({-1: 0, 0: 1, 1: 2})

# Separate features and target variable
X_train = train_dum.drop(['outcome'], axis=1)
y_train = train_dum['outcome']

# Load test data
test_df = pd.read_csv(test_file)

# Preprocess test data
test_df['clientID'] = test_df['clientID'].astype('category')
test_df['resourceID'] = test_df['resourceID'].astype('category')
test_dum = pd.get_dummies(test_df)

# Separate features and target variable for test data
X_test = test_dum.drop(['outcome', 'interactionID'], axis=1)  # Ensure 'outcome' and 'interactionID' are dropped
y_test = test_dum['outcome']

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on test data
predictions = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optionally, print confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cnf_matrix)
