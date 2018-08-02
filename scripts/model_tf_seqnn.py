#
# ─── SETUP ──────────────────────────────────────────────────────────────────────
#

import pandas as pd
import numpy as np
import udf
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return keras.backend.sqrt(
        keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.plot(
        history.epoch, np.array(history.history['rmse']), label='Train Loss')
    plt.plot(
        history.epoch, np.array(history.history['val_rmse']), label='Val loss')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
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

    # Build random Sequential model

    model = keras.Sequential([
        keras.layers.Dense(
            64, activation=tf.nn.relu, input_shape=(X_train.shape[1], )),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(1)])
        
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=[rmse])
    
    model.summary()

    # Display training progress by printing a single dot for each completed epoch.
    EPOCHS = 1000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    # Store training stats
    history = model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop])

    plot_history(history)

    #
    # ─── SUBMISSION ─────────────────────────────────────────────────────────────────
    #

    X_test['SalePrice'] = pd.DataFrame(
        model.predict(X_test.drop(columns=['Id'],
                                  axis=1)).flatten()).apply(lambda x: 10**x)
    Y_test = X_test[['Id', 'SalePrice']]
    Y_test.to_csv(path_or_buf=(data_dir + 'submit.csv'), index=False)
    print('submit.csv created!')

