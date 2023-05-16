import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('ResultTrain_NonFixedData.csv')
test_df = pd.read_csv('ResultTest_NonFixedData.csv')

X_train = train_df.drop(['Concentration', 'Photo Name'], axis=1)
y_train = train_df[train_df.columns[1]].astype('int')

X_test = test_df.drop(['Concentration', 'Photo Name'], axis=1)
y_test = test_df[test_df.columns[1]].astype('int')

cbc = CatBoostClassifier()

catboost_params = {
    "max_depth": [1, 2, 3, 5, 7, 9],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],
    'iterations': [5, 10, 50, 100, 1000, 5000],
    "border_count": [5, 10, 20, 50, 100, 200, 250],
    "l2_leaf_reg": [1, 3, 5, 10, 50, 100, 150],
}

grid_search = GridSearchCV(estimator=cbc,
                           param_grid=catboost_params,
                           n_jobs=-1,
                           cv=3,
                           scoring='accuracy',
                           error_score=0)
grid_result = grid_search.fit(X_train, y_train)
final_model = cbc.set_params(
    **grid_result.best_params_
)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(cbc.score(X_test, y_test))

with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
