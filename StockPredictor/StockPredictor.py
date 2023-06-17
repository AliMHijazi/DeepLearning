# Stock predictor that downloads or uses the previously downloaded full set of closing price data 
# for the symbol indicated, then uses torch to train and test a model to predict the next set 
# of price data. Very hard to fine tune and doesn't work great. Requires an Alphavantage API. 
# To Do: - Create a prediction projection beyond the testing date, but before the current date to 
# view predictions against actual stock data.
# - Save the prediction data to a csv in order to log it against actual stock data
# - Change the x-axis to show the date instead of day.
# - Create an accompanying script to run this program with different training/testing parameters to 
#   find the best match. Kind of like the model testing but with additional transparent steps.
# - Update model architecture to LSTM or GRU

import os
import numpy as np
import pandas as pd
import torch
import shutil
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests

symbol = 'INTC'
model_file = f'{symbol}_Model.pth'
backup_model_file = f'{symbol}_Model_Backup.pth'
api_key = 'YOUR_API_KEY'
csv_file = f'{symbol}.csv'
training_epochs = 50
training_start = '2021-03-01'
training_end = '2023-01-01'
testing_start = '2023-01-01'
testing_end = '2023-06-09'
num_days = 252 # Number of days to use in the next prediction. 252 trading days per year
num_neurons = 100
num_days_to_predict = 20

def get_data(symbol, api_key, csv_file):
    if os.path.exists(csv_file):
        print(f'Reading data from local CSV file: {csv_file}')
        data = pd.read_csv(csv_file, parse_dates=['date'], index_col='date')
    else:
        function = 'TIME_SERIES_DAILY_ADJUSTED'
        output_size = 'full'
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={output_size}&apikey={api_key}'
        print('Making API request to Alpha Vantage...')
        response = requests.get(url)
        data = response.json()
        time_series_data = data['Time Series (Daily)']
        dates = []
        close_prices = []
        for date, values in time_series_data.items():
            dates.append(date)
            close_prices.append(float(values['5. adjusted close']))
        data = pd.DataFrame({'date': dates, 'close': close_prices})
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        data = data.sort_index(ascending=True)
        print(f'Saving data to local CSV file: {csv_file}')
        data.to_csv(csv_file)
    return data

def split_data(data, training_start, training_end, testing_start, testing_end):
    data_training = data[(data.index >= training_start) & (data.index < training_end)].copy()
    data_test = data[(data.index >= testing_start) & (data.index < testing_end)].copy()
    y_test = data_test['close'].values
    return data_training, data_test, y_test

def scale_data(data):
    scaler = MinMaxScaler()
    data[['close']] = scaler.fit_transform(data[['close']])
    return scaler

def prepare_training_data(data, num_days):
    X_train = []
    y_train = []
    for i in range(num_days, data.shape[0]):
        X_train.append(data['close'].iloc[i-num_days:i].values)
        y_train.append(data['close'].iloc[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

def train_model(model, X_train, y_train, training_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    print('Training model...')
    prev_loss = float('inf')
    losses = [] 
    for epoch in range(training_epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            x = torch.tensor(X_train[i], dtype=torch.float32)
            y = torch.tensor(y_train[i], dtype=torch.float32).unsqueeze(0)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(epoch_loss / len(X_train))
        print(f'Epoch {epoch+1}: Loss={epoch_loss/len(X_train)}')
        if abs(prev_loss - epoch_loss) < 1e-5:
            print('Loss stopped decreasing significantly, stopping training early')
            break
        prev_loss = epoch_loss
    return losses 

def prepare_test_data(data_training, data_test, scaler, num_days):
    past_num_days = data_training.tail(num_days)
    df = pd.concat([past_num_days, data_test], axis=0, ignore_index=True)
    inputs = scaler.transform(df[['close']])
    X_test = []
    y_test = []
    df = pd.concat([past_num_days, data_test], axis=0, ignore_index=True)
    for i in range(num_days, inputs.shape[0]):
        X_test.append(inputs[i-num_days:i])
        y_test.append(inputs[i])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.squeeze(X_test)
    return X_test
data = get_data(symbol, api_key, csv_file)
data_training, data_test, y_test = split_data(data, training_start, training_end, testing_start, testing_end)
scaler = MinMaxScaler()
scaler.fit(data_training[['close']])
X_train, y_train = prepare_training_data(data_training, num_days)
X_test = prepare_test_data(data_training, data_test, scaler, num_days)
print(f'Obtained {len(data)} rows of data')
print(f'Splitting data into {len(data_training)} rows for training and {len(data_test)} rows for testing')
scaler = scale_data(data_training)
X_train, y_train = prepare_training_data(data_training, num_days)
model = nn.Sequential(
    nn.Linear(num_days, num_neurons),
    nn.ReLU(),
    nn.Linear(num_neurons, 1)
)
train_new_model = True
if os.path.exists(model_file):
    choice = input(f'Model file {model_file} already exists. Use existing model (y/n)? ')
    if choice.lower() == 'n':
        shutil.copyfile(model_file, backup_model_file)
        print(f'Created backup of existing model: {backup_model_file}')
        train_new_model = True
    else:
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        train_new_model = False
if train_new_model:
    losses = train_model(model, X_train, y_train, training_epochs)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
future_predictions = np.zeros(num_days_to_predict)
last_window = X_test[-1].reshape(1, -1)
for i in range(num_days_to_predict):
    future_pred = model(torch.tensor(last_window, dtype=torch.float32)).detach().numpy()
    future_pred_rescaled = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()[0]
    future_predictions[i] = future_pred_rescaled
    last_window = np.roll(last_window, -1)
    last_window[0][-1] = future_pred

X_test = prepare_test_data(data_training, data_test, scaler, num_days)
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
future_predictions = np.zeros(num_days_to_predict)
last_window = X_test[-1].reshape(1, -1)
for i in range(num_days_to_predict):
    future_pred = model(torch.tensor(last_window, dtype=torch.float32)).detach().numpy()
    future_pred_rescaled = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()[0]
    future_predictions[i] = future_pred_rescaled
    last_window = np.roll(last_window, -1)
    last_window[0][-1] = future_pred
plt.plot(y_test, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.plot(np.arange(len(y_test), len(y_test) + num_days_to_predict), future_predictions, label='Future Predicted')
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()
print(f'Predicted stock prices for next {num_days_to_predict} days: \n{np.round(future_predictions, 2)}')
