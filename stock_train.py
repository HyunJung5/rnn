import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('B.csv')
df = df[df['Close'].notnull()]
print(df.head())

df_close = df[['Close']]
df_close.index = df['Date']
print(df_close)

df_close.plot(subplots=True)
plt.show()

stock_data = df_close.values
TRAIN_SPLIT = int(len(stock_data)*0.7)

stock_data_min = stock_data.min()
stock_data_max_min_sub = stock_data.max()-stock_data.min()

stock_data = (stock_data-stock_data_min)/stock_data_max_min_sub

def data_geenrator(dataset, start_index, end_index, history_size, target_size, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):

        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        if single_step:
            labels.append(dataset[i + target_size])
        else:
            labels.append(dataset[i:i + target_size])
    return np.array(data), np.array(labels).squeeze()

past_history = 365
future_target = 30

x_train, y_train = data_geenrator(stock_data, 0, TRAIN_SPLIT,
                                           past_history,
                                           future_target)
x_val, y_val = data_geenrator(stock_data, TRAIN_SPLIT, None,
                                       past_history,
                                       future_target)


print('Single window of past history')
print(x_train.shape)
print('\n Target temperature to predict')
print(y_train.shape)

def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps

def multi_step_plot(history, true_future, prediction, title):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.title(title)

  plt.plot(num_in, np.array(history), label='History')
  plt.plot(np.arange(num_out), np.array(true_future), label='True Future')
  if prediction.any():
    plt.plot(np.arange(len(prediction)), np.array(prediction), label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

multi_step_plot(x_train[0], y_train[0], np.array([0]), 'Sample Example')

BATCH_SIZE = 32
BUFFER_SIZE = 1000

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_tensor = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_tensor = val_tensor.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True,input_shape=x_train.shape[-2:]),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(future_target)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_tensor.take(1):
    print(simple_lstm_model.predict(x).shape)

EPOCHS = 10

history = simple_lstm_model.fit(train_tensor, epochs=EPOCHS,
                      steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                      validation_data=val_tensor,
                      validation_steps=int(len(x_val)/BATCH_SIZE),
                      verbose=2)

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

plot_train_history(history, 'Training and validation loss')
simple_lstm_model.save('stock_rnn')

for x, y in val_tensor.take(3):
    plot = multi_step_plot(x[0].numpy(), y[0].numpy(),
                           simple_lstm_model.predict(x)[0], 'Validation Predict')


