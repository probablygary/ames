#
# ─── SETUP ──────────────────────────────────────────────────────────────────────
#

import pandas as pd
import numpy as np
import udf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


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


# Data folder
data_dir = './data/'

# Read input files
X = pd.read_csv(data_dir + 'clean_train.csv')
Y = X['SalePrice_log']
X.drop(labels='SalePrice_log', axis=1, inplace=True)

# Split into train, test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, shuffle=True, random_state=5)

#
# ─── BUILD MODEL ────────────────────────────────────────────────────────────────
#

# Build Multilayer Perceptron
inputs = keras.Input(shape=(x_train.shape[1], ))
hidden = keras.layers.Dense(
    units=x_test.shape[1],
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
hidden = keras.layers.Dense(
    units=x_train.shape[1] * 2, activation='relu')(hidden)
hidden = keras.layers.Dense(units=x_test.shape[1], activation='relu')(hidden)
outputs = keras.layers.Dense(units=1, activation='relu')(hidden)
tf_mlp = keras.Model(inputs=inputs, outputs=outputs)

# View model stats
tf_mlp.summary()

# Compile model
optimizer = tf.train.RMSPropOptimizer(0.001)
tf_mlp.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=[rmse])

# Train model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)
EPOCHS = 500
history = tf_mlp.fit(
    x=x_train,
    y=y_train,
    epochs=EPOCHS,
    verbose=1,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stop])

# Evaluate model
[loss, error] = tf_mlp.evaluate(x=x_test, y=y_test, verbose=1)
print("Testing set RMSE:\t{}".format(error))
udf.log_result('./logs/', 'train_log.csv', x_train.columns,
               tf_mlp.get_config(), error)
plot_history(history)

#
# ─── SUBMISSION ─────────────────────────────────────────────────────────────────
#

X_submit = pd.read_csv(data_dir + 'clean_test.csv')
X_submit['SalePrice'] = pd.DataFrame(
    tf_mlp.predict(X_submit.drop(columns=['Id'],
                                 axis=1)).flatten()).apply(lambda x: 10**x)
Y_submit = X_submit[['Id', 'SalePrice']]
Y_submit.to_csv(path_or_buf=(data_dir + 'submit.csv'), index=False)
print('submit.csv created!')
