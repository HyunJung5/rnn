import numpy as np
import pandas as pd
import cv2
import tflite_runtime.interpreter as tflite
from matplotlib import pyplot as plt
import time

ans = input('do you wanna check running time? (y/n)')
start = time.perf_counter()

interpreter = tflite.Interpreter(model_path="./converted_model.tflite")
interpreter.allocate_tensors()

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
stock_data_max_min_sub = stock_data.max() - stock_data.min()
stock_data = (stock_data - stock_data_min) / stock_data_max_min_sub


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


past_history = 365
future_target = 30

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array([stock_data[-past_history:]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

if ans == 'y' or ans == 'Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)


plt.plot(output_data, 'o-')
plt.ylabel('Prediction')
plt.savefig('savefig_default.png')
img = cv2.imread('./savefig_default.png')
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()