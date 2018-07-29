#
# ─── SETUP ──────────────────────────────────────────────────────────────────────
#

import pandas as pd
import numpy as np
import udf
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Data folder
data_dir = './data/'

# Read input files
X_train = pd.read_csv(data_dir + 'clean_train.csv')
Y_train = X_train['SalePrice_log']
X_train.drop(labels='SalePrice_log', axis=1, inplace=True)
X_test = pd.read_csv(data_dir + 'clean_test.csv')

#
# ─── BUILD MODEL ────────────────────────────────────────────────────────────────
#

if __name__ == '__main__':

    # Build random forest regressor,
    # Optimise hyperparameters with GridSearchCV
    param_grid = {
        'hidden_layer_sizes': [i for i in range(5, 50, 5)],
        'activation': ['relu', 'logistic'],
        'learning_rate': ['constant'],
        'learning_rate_init': [0.0001],
        'alpha': [0.0001],
        'max_iter': [10000],
        'early_stopping': [True],
    }

    mlpr = MLPRegressor(random_state=5)
    rmse_scorer = make_scorer(udf.rmse, greater_is_better=False)

    mlpr_cv = GridSearchCV(
        n_jobs=-1,
        estimator=mlpr,
        param_grid=param_grid,
        cv=10,
        verbose=3,
        return_train_score=True,
        scoring=rmse_scorer)

    mlpr_cv.fit(X_train, Y_train)
    print('Best parameters: ' + str(mlpr_cv.best_params_))
    print('Error: ' + str(abs(mlpr_cv.best_score_)))

    # Log Training
    udf.log_result('./logs/', 'train_log.csv', X_train.columns,
                   mlpr_cv.best_params_, abs(mlpr_cv.best_score_))

    #
    # ─── SUBMISSION ─────────────────────────────────────────────────────────────────
    #

    X_test['SalePrice'] = pd.DataFrame(
        mlpr_cv.predict(X_test.drop(labels='Id',
                                   axis=1))).apply(lambda x: 10**x)
    Y_test = X_test[['Id', 'SalePrice']]
    Y_test.to_csv(path_or_buf=(data_dir + 'submit.csv'), index=False)
    print('submit.csv created!')