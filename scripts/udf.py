# User-defined functions for this project
# A little messy 'cos it's just for personal convenience
import pandas as pd
import numpy as np
import os
import sys
import json
from time import asctime
from sklearn import preprocessing
from scipy import stats


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


def rmse(y, y_pred):
    """
    Define root-mean-squared error for model optimisation.
    """
    return np.sqrt(np.mean(np.square(y - y_pred)))


def check_dir(path, filename):
    """
    Check directory for presence of a file.

    Args:
        path: String of folder path.
        filename: String of filename with extension.

    Returns:
        Boolean.
    """

    flag = False
    with os.scandir(path) as path:
        for entry in path:
            if os.DirEntry.is_file(entry) and entry.name == filename:
                flag = True
    return flag


def log_result(log_dir, log_name, features, params, score):
    """
    Log results of model training to a .csv file.

    Args:
        log_dir: String containing directory of log file, ending in '/'.
        log_name: String of log file name with extension.
        features: List of names of features used in training.
        params: Dict of parameters used for the model.
        score: Training score as int/float.
    
    Returns None.
    """

    model = os.path.basename(sys.argv[0])

    if check_dir(log_dir, log_name):
        log = pd.read_csv(log_dir + log_name, index_col='datetime')
        log = log.append(
            pd.DataFrame.from_records(
                data={
                    'datetime': [asctime()],
                    'model':
                    model,
                    'features':
                    ', '.join(str(feat) for feat in features),
                    'params':
                    ', '.join("{!s}={!r}".format(key, val)
                              for (key, val) in params.items()),
                    'score':
                    score
                },
                index='datetime'),
            sort=True)

    else:
        log = pd.DataFrame.from_records(
            data={
                'datetime': [asctime()],
                'model':
                model,
                'features':
                ', '.join(str(feat) for feat in features),
                'params':
                ', '.join("{!s}={!r}".format(key, val)
                          for (key, val) in params.items()),
                'score':
                score
            },
            index='datetime')

    log.to_csv(path_or_buf=(log_dir + log_name))

    return None