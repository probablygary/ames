# -- Setup --
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer


# UDFs
def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y)**2))


# Data folder
data_dir = './data/'

# Read input files
X_train = pd.read_csv(data_dir + 'clean_train.csv')
Y_train = X_train['SalePrice_log']
X_train.drop(labels='SalePrice_log', axis=1, inplace=True)
X_test = pd.read_csv(data_dir + 'clean_test.csv')
# ----

# -- Build Model --
# Build random forest regressor,
# Optimise hyperparameters with GridSearchCV
param_grid = {
    'n_estimators': [100, 250, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 5, 10, 25, 50],
}
rfr = RandomForestRegressor(random_state=5)
rmse_scorer = make_scorer(rmse, greater_is_better=False)
rfr_cv = GridSearchCV(
    estimator=rfr,
    param_grid=param_grid,
    cv=10,
    verbose=3,
    return_train_score=True,
    scoring=rmse_scorer)

# rfr_cv.fit(X_train, Y_train)
# print(rfr_cv.best_params_)

# x_train = X_train.loc[train_idx]
# y_train = Y_train.loc[train_idx]
# x_val = X_train.loc[val_idx]
# y_val = Y_train.loc[val_idx]
# rfr.fit(x_train, y_train)
# predictions = rfr.predict(x_val)
# print('{} RMSE: {:<5f}'.format(i + 1, mean_squared_error(
#     y_val, predictions)))

# fi = pd.DataFrame(data={
#     'feature': X_train.columns,
#     'importance': rfr.feature_importances_
# })
# print('\nFeature Importance:')
# print(fi.reindex(fi.importance.sort_values(ascending=False).index))

# # -- Create .CSV for submission --
rfr = RandomForestRegressor(random_state=5, max_depth=None, max_features='sqrt', n_estimators=500)
rfr.fit(X_train, Y_train)
X_test['SalePrice'] = np.exp(rfr.predict(X_test.drop(labels='Id', axis=1)))
submit = X_test[['Id', 'SalePrice']]
submit.to_csv(path_or_buf=(data_dir + 'submit.csv'), index=False)