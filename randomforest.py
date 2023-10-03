#AMZN 7:2:1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.7 * len(df1))
test_size = int(0.2 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 13. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('AMZN.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()

###
#AMZN 6:3:1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.6 * len(df1))
test_size = int(0.3 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('AMZN.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()
###
#AMZN 6:2:2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.6 * len(df1))
test_size = int(0.2 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('AMZN.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()

###
#BABA 7:2:1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('BABA.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.7 * len(df1))
test_size = int(0.2 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)


# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('BABA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()

###
#BABA 6:3:1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.6 * len(df1))
test_size = int(0.3 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('BABA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()
###
#BABA 6:2:2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.6 * len(df1))
test_size = int(0.2 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('BABA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()

####
#EBAY 7:2:1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.7 * len(df1))
test_size = int(0.2 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('EBAY.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()

###
#EBAY 6:3:1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.6 * len(df1))
test_size = int(0.3 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('EBAY.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()

###
#EBAY 6:2:2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Đọc file csv và gắng index với giá Close
df = pd.read_csv('AMZN.csv')
df1 = df.reset_index()['Close']

# 3. Scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# 4. Chia train test
train_size = int(0.6 * len(df1))
test_size = int(0.2 * len(df1))
val_size = len(df1) - train_size - test_size

train_data = df1[:train_size]
test_data = df1[train_size:train_size+test_size]
val_data = df1[train_size+test_size:]

# 5. Hàm Create Dataset
import numpy

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 6. Reshape into X=t,t+1,t+2..t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 7. Định nghĩa và huấn luyện mô hình Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Dự đoán giá trị train, test và validation
train_predict = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)


# 9. Chuẩn hóa dữ liệu y_pred, y_pred_val
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1, 1))

# 10. Đánh giá độ chính xác
import numpy as np

# Tính RMSE
valid_rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print('Validation RMSE:', valid_rmse)
print('Testing RMSE:', test_rmse)

# Tính MSE
valid_mse = np.mean((y_pred_val - y_val)**2)
test_mse = np.mean((y_pred - y_test)**2)
print('Validation MSE:', valid_mse)
print('Testing MSE:', test_mse)

# Tính MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
valid_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print('Validation MAPE:', valid_mape)
print('Testing MAPE:', test_mape)
# 11. Vẽ biểu đồ
from matplotlib import dates
df = pd.read_csv('EBAY.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df['Close']

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%m-%Y'))
# Xoay và căn chỉnh nhãn trục x
plt.gcf().autofmt_xdate()

plt.plot(df1.index[:train_size], scaler.inverse_transform(train_data))
plt.plot(df1.index[train_size:train_size + test_size], scaler.inverse_transform(test_data))
plt.plot(df1.index[train_size+101:train_size + test_size], y_pred[:test_size - time_step, 0])
plt.plot(df1.index[train_size + test_size:train_size + test_size + val_size], scaler.inverse_transform(val_data))
plt.plot(df1.index[train_size + test_size+101:train_size + test_size + val_size], y_pred_val)

plt.legend(['Train', 'Test', 'Predict', 'Validate', 'ValidatePred'])
plt.show()