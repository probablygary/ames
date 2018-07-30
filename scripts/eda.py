# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats


# -- Setup --
def anova(df, dependent_var, independent_var):
    """
    Conduct one-way ANOVA of categorical features.

    Args:
        df: Pandas DataFrame containing both dependent and independent features.
        dependent_var: name of a multi-level dependent feature contained in df.
        independent_vars: name of a numerical independent feature contained in df.
    
    Returns:
        f-value and p-value of ANOVA of the dependent feature.
    """

    values = list()
    df.fillna('missing', inplace=True)
    for level in df[dependent_var].unique().tolist():
        values.append(df.loc[df[dependent_var] == level, independent_var]
                      .values.tolist())
    f_value, p_value = stats.f_oneway(*values)

    return f_value, p_value


def label_encode(var, values):
    """
    Label encode ordinal features from string values to numerical values.

    Args:
        var: Pandas DataFrame or Series to be encoded.
        values: List of labels in ascending order.
    
    Returns:
        Pandas DataFrame of transformed feature.
    """

    le = preprocessing.LabelEncoder()
    le.fit(values)
    var_encoded = le.transform(var)

    return var_encoded
# Seaborn settings
sns.set()

# Data folder
data_dir = './data/'

# Read input files
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
print('Setup complete')
# ----

# -- Spelunking --
# %%
# Summary stats
train.info()
train.head(5)
train.describe()

# %%
# Count NA
na_count = train.isnull().sum().sort_values(ascending=False)
na_pct = (train.isnull().sum() / train.isnull().count() *
          100).sort_values(ascending=False)
missing = pd.concat([na_count, na_pct], axis=1, keys=['Total', '%'])
missing.head(20)

# %%
# Target Feature
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
sns.distplot(train['SalePrice'])

# %%
# Log-transform target feature
train['SalePrice_log'] = np.where(train['SalePrice'] <= 0, train['SalePrice'],
                                  np.log10(train['SalePrice']))
Y = train['SalePrice_log']
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
sns.distplot(train['SalePrice'], ax=ax[0])
sns.distplot(train['SalePrice_log'], ax=ax[1])
plt.show()
# %%
print('D\'Agostino-Pearson Test\n{}'.format('-' * 10))
s, p_value = stats.normaltest(train['SalePrice'])
print('SalePrice \t Stat: {},\tp-value: {}'.format(s, p_value))
s, p_value = stats.normaltest(train['SalePrice_log'])
print('SalePrice_log \t Stat: {},\tp-value: {}'.format(s, p_value))
# %%
#  Correlations
corr = train.drop(labels='SalePrice', axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 1, as_cmap=True)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=.3,
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={'shrink': .5})
plt.show()
print('Correlation with \'SalePrice_log\':')
print(corr['SalePrice_log'].sort_values(ascending=False))

# %%
# ANOVA of categorical variables
anova_output = pd.DataFrame(columns=['feature', 'f_value', 'p_value'])
for col in train.select_dtypes(include=['object']).columns.tolist():
    f_value, p_value = anova(train, col, 'SalePrice_log')
    anova_output = anova_output.append(
        {
            'feature': col,
            'f_value': f_value,
            'p_value': p_value
        },
        ignore_index=True)
print(
    anova_output.reindex(
        anova_output.p_value.sort_values(ascending=True).index))

# -- Feature engineering --
# Initialise DataFrame for independent variables
# %%
X = pd.DataFrame()

# %%
# Label-encoding ordinal features
grades = ['missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
label_vars = ['ExterQual', 'BsmtQual', 'KitchenQual']

for var in label_vars:
    X[var + '_enc'] = label_encode(train[var].fillna('missing'), grades)

# %%
# One-hot encoding discrete categorical features
X = pd.merge(
    pd.get_dummies(train['Neighborhood'], drop_first=True),
    X,
    left_index=True,
    right_index=True)

# %%
# -- Project Runway --
# Build random forest regressor
kf = KFold(n_splits=10)
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    x_train = X.loc[train_idx]
    y_train = Y.loc[train_idx]
    x_test = X.loc[test_idx]
    y_test = Y.loc[test_idx]
    rfr = RandomForestRegressor(verbose=0)
    rfr.fit(x_train, y_train)
    predictions = rfr.predict(x_test)
    print('{} RMSE: {:<5f}'.format(i + 1,
                                   mean_squared_error(y_test, predictions)))

fi = pd.DataFrame(data={
    'feature': X.columns,
    'importance': rfr.feature_importances_
})
print('\nFeature Importance:')
print(fi.reindex(fi.importance.sort_values(ascending=False).index))
