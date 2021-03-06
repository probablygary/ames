#
# ─── SETUP ──────────────────────────────────────────────────────────────────────
#

import pandas as pd
import numpy as np
import udf
from sklearn import preprocessing

# Data folder
data_dir = './data/'

# Read input files
full = pd.read_csv(data_dir + 'train.csv').append(
    pd.read_csv(data_dir + 'test.csv'), sort=True, ignore_index=True)
print('Setup complete')

#
# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────────
#

# Initialise DataFrame for independent variables
X = pd.DataFrame(full['Id'])

# Log-transform target feature
X['SalePrice_log'] = np.where(full['SalePrice'] <= 0, full['SalePrice'],
                              np.log10(full['SalePrice']))

# Label-encode ordinal features
grades = ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
label_vars = ['ExterQual', 'BsmtQual', 'KitchenQual']

for var in label_vars:
    X[var + '_enc'] = udf.label_encode(full[var].fillna('missing'), grades)

# One-hot encode discrete categorical features
X = pd.merge(
    pd.get_dummies(full['Neighborhood'], drop_first=True),
    X,
    left_index=True,
    right_index=True)

# Ordinal features
ord_vars = ['OverallQual']
for var in ord_vars:
    X[var] = full[var]

# Numerical features
num_vars = ['GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'TotRmsAbvGrd', 'LotArea          ']
for var in num_vars:
    X[var + '_scaled'] = preprocessing.scale(full[var].fillna(0))

#
# ─── SAVE DATA ──────────────────────────────────────────────────────────────────
#

X.loc[X['Id'] <= 1460].drop(
    labels='Id', axis=1).to_csv(
        path_or_buf=data_dir + 'clean_train.csv', index=False)
X.loc[X['Id'] > 1460].drop(
    labels='SalePrice_log', axis=1).to_csv(
        path_or_buf=data_dir + 'clean_test.csv', index=False)

print(X.info())
# ----