import os
import numpy as np
import pandas as pd
import torch
import shutil
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import requests

symbol = 'NVDA'
model_file = f'{symbol}_Model.pth'
backup_model_file = f'{symbol}_Model_Backup.pth'
api_key = 'YOUR_API_KEY'
csv_file = f'{symbol}.csv'
training_epochs = 100
training_start = '2021-01-01'
training_end = '2023-06-01'
testing_start = '2023-01-01'
testing_end = '2023-06-05'
num_days = 30 # Number of days to use in the next prediction.
num_days_to_predict = 5

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


def split_data(data, training_start_date, training_end_date, test_start_date, test_end_date):
    data_training = data[(data.index >= training_start_date) & (data.index < training_end_date)].copy()
    data_test = data[(data.index >= test_start_date) & (data.index < test_end_date)].copy()
    return data_training, data_test

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
        print(f'Epoch {epoch+1}: Loss={epoch_loss/len(X_train)}')
        if abs(prev_loss - epoch_loss) < 1e-5:
            print('Loss stopped decreasing significantly, stopping training early')
            break
        prev_loss = epoch_loss

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

def predict_next_day(model, last_num_days, scaler):
    inputs = last_num_days
    inputs = torch.tensor(inputs, dtype=torch.float32)
    predicted_price = model(inputs).item()
    return predicted_price

training_start_date = training_start
training_end_date = training_end
test_start_date = testing_start
test_end_date = testing_end
data = get_data(symbol, api_key, csv_file)
data_training, data_test = split_data(data, training_start_date, training_end_date, test_start_date, test_end_date)
scaler = scale_data(data_training)
X_train, y_train = prepare_training_data(data_training, num_days)
X_test = prepare_test_data(data_training, data_test, scaler, num_days)
#data = get_data(symbol, api_key, csv_file)
print(f'Obtained {len(data)} rows of data')
data_training, data_test = split_data(data, training_start, training_end, testing_start, testing_end)
print(f'Splitting data into {len(data_training)} rows for training and {len(data_test)} rows for testing')
scaler = scale_data(data_training)
X_train, y_train = prepare_training_data(data_training, num_days)
model = nn.Sequential(
    nn.Linear(num_days, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
train_new_model = True
if os.path.exists(model_file):
    choice = input(f'Model file {model_file} already exists. Use existing model (y/n)? ')
    if choice.lower() == 'n':
        shutil.copyfile(model_file, backup_model_file)
        print(f'Created backup of existing model: {backup_model_file}')
        train_new_model = False
    else:
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        train_new_model = False

if train_new_model:
    train_model(model, X_train, y_train, training_epochs)
X_test = prepare_test_data(data_training, data_test, scaler, num_days)
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
print('Model training complete')
if not os.path.exists(model_file) or choice.lower() == 'n':
    torch.save(model.state_dict(), model_file)
    
last_num_days = data.tail(num_days)
predicted_prices = []
for i in range(num_days_to_predict):
    predicted_price = predict_next_day(model, last_num_days['close'].values.reshape(1, -1), scaler)
    predicted_prices.append(round(predicted_price, 2))
    last_num_days = pd.concat([last_num_days.iloc[1:], pd.DataFrame({'close': [predicted_price]})], ignore_index=True)
print(f'Predicted stock prices for next {num_days_to_predict} days: {predicted_prices}')

