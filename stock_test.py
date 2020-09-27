import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

ans = input('do you wanna check running time? (y/n)')
start = time.perf_counter()

df = pd.read_csv('B.csv')
df = df[df['Close'].notnull()]
print(df.head())
df_close = df[['Close']]
df_close.index = df['Date']
print(df_close)
df_close.plot(subplots=True)
if ans == 'n' or ans == 'N':
    plt.show()

stock_data = df_close.values
stock_data_min = stock_data.min()
stock_data_max_min_sub = stock_data.max()-stock_data.min()
stock_data = (stock_data-stock_data_min)/stock_data_max_min_sub

past_history = 365
future_target = 30

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
  if ans == 'n' or ans == 'N':
      plt.show()

simple_lstm_model = tf.keras.models.load_model('stock_rnn', compile = False)

multi_step_plot(stock_data[-past_history:], np.array([0]),
                simple_lstm_model.predict(np.array([stock_data[-past_history:]]))[0],
                'Future month Predict')

if ans == 'y' or ans == 'Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
