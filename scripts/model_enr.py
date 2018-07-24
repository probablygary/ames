# -- Setup --
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV


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
# Build ElasticNet regressor,
# Optimise hyperparameters with CV
enr_cv = ElasticNetCV(
    random_state=5,
    cv=5,
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    fit_intercept=True,
    precompute='auto',
    copy_X=True,
    verbose=2)

enr_cv.fit(X_train, Y_train)
print(enr_cv.alpha_)
print(enr_cv.mse_path_)

# -- Create .CSV for submission --
X_test['SalePrice'] = pd.DataFrame(
    enr_cv.predict(X_test.drop(labels='Id', axis=1))).apply(lambda x: 10**x)
Y_test = X_test[['Id', 'SalePrice']]
Y_test.to_csv(path_or_buf=(data_dir + 'submit.csv'), index=False)
print('submit.csv created!')