# -- Setup --
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# UDFs
def rmse(y, y_pred):
    '''
    Defining root-mean-squared error for 
    model optimisation
    '''
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
# Build gradient boosted regressor,
# Optimise hyperparameters with GridSearchCV
param_grid = {
    'learning_rate': [0.005, 0.01, 0.02],
    'n_estimators': [1000, 1250, 1500, 1750],
    'max_depth': [1, 3, 5],
    'max_features': ['log2']
}
gbr = GradientBoostingRegressor(random_state=5)
rmse_scorer = make_scorer(rmse, greater_is_better=False)
gbr_cv = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    return_train_score=True,
    scoring=rmse_scorer)

gbr_cv.fit(X_train, Y_train)
print(gbr_cv.best_params_)
print(gbr_cv.best_score_)

# Feature Importance
fi = pd.DataFrame(
    data={
        'Feature': X_train.columns,
        'Importance': gbr_cv.best_estimator_.feature_importances_
    })
print('\nFeature Importance:')
print(fi.reindex(fi.Importance.sort_values(ascending=False).index))

# -- Create .CSV for submission --
X_test['SalePrice'] = pd.DataFrame(
    gbr_cv.predict(X_test.drop(labels='Id', axis=1))).apply(lambda x: 10**x)
Y_test = X_test[['Id', 'SalePrice']]
Y_test.to_csv(path_or_buf=(data_dir + 'submit.csv'), index=False)
print('submit.csv created!')