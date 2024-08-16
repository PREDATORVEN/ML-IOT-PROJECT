import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
# %matplotlib inline

outcomes_df = pd.read_csv(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\train_outcomes.csv")

print(outcomes_df.shape)
outcomes_df.head()

outcomes_df['clientID']  = outcomes_df['clientID'].astype('category')
outcomes_df['resourceID']  = outcomes_df['resourceID'].astype('category')

outcomes_dum= pd.get_dummies(outcomes_df)
outcomes_dum.head()

X_train, X_test, y_train, y_test = train_test_split(outcomes_dum.drop(['outcome'], axis=1),
                                                    outcomes_dum['outcome'], test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train) + 1
y_test = np.array(y_test) + 1

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest= xgb.DMatrix(X_test,label=y_test)

watchlist  = [(dtest,'eval'), (dtrain,'train')]
param = {'max_depth':10, 'eta':0.3, 'silent':1, 'gamma':5,'objective':'multi:softprob','num_class':3 }
num_round = 50
bst = xgb.train(param, dtrain, num_round,watchlist)

preds = bst.predict(dtest)
predictions = np.argmax(preds,axis=1)
cnf_matrix = confusion_matrix(y_test, predictions)
print(cnf_matrix)
print("Accuracy_Score : ",accuracy_score(y_test,predictions))

# OUTCOMES HALFMODEL!!
'''
[[ 596 1369    9]
 [ 202 8452 1050]
 [  68 1146 7108]]
Accuracy_Score :  0.8078
'''
#FULL MODEL!!
# full_features = xgb.DMatrix(outcomes_dum.drop(['outcome'], axis=1), label=np.array(outcomes_dum['outcome'])+1)

# num_round = 100
# bstfull = xgb.train(param, full_features, num_round)

# bstfull.save_model(r'D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\model\xgb_full_model.model')

bstfull = xgb.Booster() #init model
# bstfull.load_model(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\model\xgb_full_model.model") # load data
bstfull.load_model(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\model\xgbfull.model") # load data

testing_df = pd.read_csv(r"D:\ml-iot-resource_(1)[1] (1)\ml-iot-resource-allocation-master\data\test_outcomes.csv")

print(testing_df.shape)
testing_df.head()

testing_df['clientID']  = testing_df['clientID'].astype('category')
testing_df['resourceID']  = testing_df['resourceID'].astype('category')
print(testing_df['clientID'].cat.categories )
print(testing_df['resourceID'].cat.categories )

testing_dum= pd.get_dummies(testing_df)
testing_dum.head()

full_test = xgb.DMatrix(testing_dum.drop(['outcome','interactionID'], axis=1), label=np.array(testing_dum['outcome'])+1)
pred = bstfull.predict(full_test)

predictions = np.argmax(pred,axis=1)
cnf_matrix = confusion_matrix(np.array(testing_dum['outcome'])+1, predictions)
accuracy_src=accuracy_score(np.array(testing_dum['outcome'])+1, predictions)
print(cnf_matrix)
print("Accuracy_Score :",accuracy_src)

