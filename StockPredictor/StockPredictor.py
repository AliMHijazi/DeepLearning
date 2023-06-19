# Stock predictor that downloads or uses the previously downloaded full set of closing price data 
# for the symbol indicated, then uses torch to train and test a model to predict the next set 
# of price data. Very hard to fine tune and doesn't work great. Requires an Alphavantage API. 
# To Do: - Create a prediction projection beyond the testing date, but before the current date to 
# view predictions against actual stock data.
# - Save the prediction data to a csv in order to log it against actual stock data
# - Create an accompanying script to run this program with different training/testing parameters to 
#   find the best match. Kind of like the model testing but with additional transparent steps.
# - Compare today's date with the latest date in the stock data file and prompt user to download new data
#   if there is more available than what is currently downloaded. 
# - Update model architecture to LSTM or GRU

import os
import numpy as np
import pandas as pd
import torch
import shutil
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests

DJI_normalize = 0
NASDAQ_normalize = 1
symbol = 'TSLA'
model_file = f'{symbol}_Model.pth'
backup_model_file = f'{symbol}_Model_Backup.pth'
api_key = 'YOUR_API_KEY'
csv_file = f'{symbol}.csv'
training_epochs = 30
training_start = '2013-01-01'
training_end = '2021-01-01'
testing_start = '2021-01-01'
testing_end = '2023-06-09'
num_days = 100 # Number of days to use in the next prediction. 252 trading days per year
num_neurons = 100
num_days_to_predict = 100

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone().detach()
        self.y = y.clone().detach()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data(symbol, api_key, csv_file):
    if os.path.exists(csv_file):
        print(f'Reading data from local CSV file: {csv_file}')
        data = pd.read_csv(csv_file, parse_dates=['Date'])
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
        data = pd.DataFrame({'Date': dates, 'Closing Price': close_prices})
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date', ascending=True)
        print(f'Saving data to local CSV file: {csv_file}')
        data.to_csv(csv_file, index=False)
    return data

def split_data(data, training_start, training_end, testing_start, testing_end):
    data_training = data[(data.index >= training_start) & (data.index < training_end)].copy()
    data_test = data[(data.index >= testing_start) & (data.index < testing_end)].copy()
    y_test = data_test['Closing Price'].values
    test_dates = data_test.index
    return data_training, data_test, y_test, test_dates
    
def prepare_training_data(data, num_days):
    X_train = []
    y_train = []
    for i in range(num_days, data.shape[0]):
        X_train.append(data['Closing Price'].iloc[i-num_days:i].values)
        y_train.append(data['Closing Price'].iloc[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train
    
def train_model(model, train_dataloader, training_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    print('Training model...')
    prev_loss = float('inf')
    losses = [] 
    for epoch in range(training_epochs):
        epoch_loss = 0
        y_pred_list = []
        y_true_list = []
        for X_batch, y_batch in train_dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_list.extend(y_pred.detach().numpy().flatten())
            y_true_list.extend(y_batch.detach().numpy().flatten())
        losses.append(epoch_loss / len(train_dataloader))
        mae = mean_absolute_error(y_true_list, y_pred_list)
        mse = mean_squared_error(y_true_list, y_pred_list)
        print(f'Epoch {epoch+1}: Loss={epoch_loss/len(train_dataloader)}, MAE={mae}, MSE={mse}')
        if abs(prev_loss - epoch_loss) < 1e-9:
            print('Loss stopped decreasing significantly, stopping training early')
            break
        prev_loss = epoch_loss
    return losses

def prepare_test_data(data_training, data_test, scaler, num_days):
    past_num_days = data_training.tail(num_days)
    df = pd.concat([past_num_days, data_test], axis=0, ignore_index=True)
    inputs = scaler.transform(df[['Closing Price']])
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
data_nasdaq = get_data('QQQ', api_key, f'NASDAQ.csv')
data_dowjones = get_data('DIA', api_key, f'DOWJONES.csv')

data.set_index('Date', inplace=True)
data_nasdaq.set_index('Date', inplace=True)
data_dowjones.set_index('Date', inplace=True)
if(DJI_normalize == 1):
    data['Closing Price'] /= data_nasdaq['Closing Price']
if(NASDAQ_normalize == 1):
    data['Closing Price'] /= data_dowjones['Closing Price']

data.index = pd.to_datetime(data.index)
data_training, data_test, y_test, test_dates = split_data(data, training_start, training_end, testing_start, testing_end)
print(f'Data range: {data.index.min()} to {data.index.max()}')
scaler = MinMaxScaler()
scaler.fit(data_training[['Closing Price']])
X_train, y_train = prepare_training_data(data_training, num_days)
X_test = prepare_test_data(data_training, data_test, scaler, num_days)
print(f'Obtained {len(data)} rows of data')
print(f'Splitting data into {len(data_training)} rows for training and {len(data_test)} rows for testing')
model = nn.Sequential(
    nn.Linear(num_days, num_neurons),
    nn.ReLU(),
    nn.Linear(num_neurons, 1)
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_dataset = StockDataset(X_train_tensor, y_train_tensor)       
batch_size = 32
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = train_model(model, train_dataloader, training_epochs)
    torch.save(model.state_dict(), model_file)
    print(f'Saved trained model to file: {model_file}')
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
if(DJI_normalize == 1):
    y_pred_rescaled *= data_dowjones.loc[data_test.index]['Closing Price']
    y_test *= data_dowjones.loc[data_test.index]['Closing Price']
if(NASDAQ_normalize == 1):
    y_pred_rescaled *= data_nasdaq.loc[data_test.index]['Closing Price']
    y_test *= data_nasdaq.loc[data_test.index]['Closing Price']
future_predictions = np.zeros(num_days_to_predict)
last_window = X_test[-1].reshape(1, -1)
for i in range(num_days_to_predict):
    future_pred = model(torch.tensor(last_window, dtype=torch.float32)).detach().numpy()
    future_pred_rescaled = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()[0]
    future_predictions[i] = future_pred_rescaled
    last_window = np.roll(last_window, -1)
    last_window[0][-1] = future_pred
dates = test_dates.to_pydatetime()
if(DJI_normalize == 1):
    last_dowjones_price = data_dowjones.iloc[-1]['Closing Price']
    future_predictions *= last_dowjones_price
if(NASDAQ_normalize == 1):
    last_nasdaq_price = data_nasdaq.iloc[-1]['Closing Price']
    future_predictions *= last_nasdaq_price
plt.plot(dates, y_test, label='Actual')
plt.plot(dates, y_pred_rescaled, label='Predicted')
future_dates = pd.date_range(start=dates[-1], periods=num_days_to_predict + 1, freq='B')[1:]
plt.plot(future_dates, future_predictions, label='Future Predicted')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gcf().autofmt_xdate()
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
print(f'Predicted stock prices for next {num_days_to_predict} days: \n{np.round(future_predictions, 2)}')
