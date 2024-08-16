import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
outcomes_df = pd.read_csv(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\train_outcomes.csv")

outcomes_df['clientID'] = outcomes_df['clientID'].astype('category')
outcomes_df['resourceID'] = outcomes_df['resourceID'].astype('category')

# Feature engineering
outcomes_df['client_resource'] = outcomes_df['clientID'].astype(str) + "_" + outcomes_df['resourceID'].astype(str)
outcomes_df['client_resource'] = outcomes_df['client_resource'].astype('category')

# One-hot encoding
outcomes_dum = pd.get_dummies(outcomes_df, columns=['clientID', 'resourceID', 'client_resource'])

# Split the data
X = outcomes_dum.drop(['outcome'], axis=1)
y = outcomes_dum['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Adjust labels
y_train = np.array(y_train) + 1
y_test = np.array(y_test) + 1

# Create DMatrix
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.2, 0.3]
}

xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Train the model with best parameters
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 200
bst = xgb.train(best_params, dtrain, num_round, watchlist, early_stopping_rounds=20)

# Make predictions
preds = bst.predict(dtest)
predictions = np.argmax(preds, axis=1)

# Evaluate the model
cnf_matrix = confusion_matrix(y_test, predictions)
print(cnf_matrix)
print("Accuracy Score:", accuracy_score(y_test, predictions))