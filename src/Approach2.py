import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

# Load the pre-trained model
bstfull = xgb.Booster() #init model
bstfull.load_model(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\model\xgbfull.model") # load data

# Load the testing data
testing_df = pd.read_csv(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\test_outcomes.csv")
print(testing_df.shape)
print(testing_df.head())

# Convert categorical columns to category dtype
testing_df['clientID'] = testing_df['clientID'].astype('category')
testing_df['resourceID'] = testing_df['resourceID'].astype('category')
print(testing_df['clientID'].cat.categories)
print(testing_df['resourceID'].cat.categories)

# Apply one-hot encoding
testing_dum = pd.get_dummies(testing_df)
print(testing_dum.head())

# Prepare the data for XGBoost
full_test = xgb.DMatrix(testing_dum.drop(['outcome', 'interactionID'], axis=1), label=np.array(testing_dum['outcome']) + 1)

# Make predictions
pred = bstfull.predict(full_test)
predictions = np.argmax(pred, axis=1)
cnf_matrix = confusion_matrix(np.array(testing_dum['outcome']) + 1, predictions)
print("Confusion Matrix:\n", cnf_matrix)

# Calculate and print accuracy
accuracy = accuracy_score(np.array(testing_dum['outcome']) + 1, predictions)
print("Accuracy:", accuracy)

# Calculate and print precision
precision = precision_score(np.array(testing_dum['outcome']) + 1, predictions, average='weighted')
print("Precision:", precision)

# Calculate and print recall
recall = recall_score(np.array(testing_dum['outcome']) + 1, predictions, average='weighted')
print("Recall:", recall)

# Calculate and print F1-score
f1 = f1_score(np.array(testing_dum['outcome']) + 1, predictions, average='weighted')
print("F1 Score:", f1)

# Load the training data for hyperparameter tuning
train_df = pd.read_csv(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\train_outcomes.csv")
train_df['clientID'] = train_df['clientID'].astype('category')
train_df['resourceID'] = train_df['resourceID'].astype('category')
train_dum = pd.get_dummies(train_df)

X = train_dum.drop(['outcome'], axis=1)
y = train_dum['outcome'] + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning using RandomizedSearchCV
param_distributions = {
    'max_depth': [6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50,100],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1]
}

xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, early_stopping_rounds=10, eval_metric='mlogloss')
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, scoring='accuracy', cv=3, verbose=2, n_jobs=-1, n_iter=10)
random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print("Best parameters found: ", random_search.best_params_)
print("Best accuracy found: ", random_search.best_score_)

# Training with the best parameters
best_params = random_search.best_params_
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

# Feature Selection
selector = SelectFromModel(xgb_model, threshold=0.01, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Re-train the model with selected features
xgb_model.fit(X_train_selected, y_train)

# Make predictions and evaluate the model
predictions = xgb_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("After Feature Selection and Hyperparameter Tuning:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
